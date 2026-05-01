"""
Spawn N Thunder Compute workers from a snapshot, inject env vars, run bootstrap.

Run on a developer laptop (us). Requires:
  - tnr CLI installed and `tnr login` done (uses ~/.thunder/token)
  - SSH private key matching the public key Thunder seeds in instances
  - boto3 / requests / paramiko

Usage:
    export DATABASE_URL='postgres://…'
    export AWS_ACCESS_KEY_ID=…
    export AWS_SECRET_ACCESS_KEY=…
    export PAIR_OUT_S3_PREFIX='s3://dr-handpose-outputs/v1/'

    python scripts/spawn.py \
        --num-workers 13 \
        --snapshot dynhamr-handpose-v1 \
        --gpu a6000 --num-gpus 1 --vcpus 8 --primary-disk 200 \
        --ssh-key ~/.ssh/id_ed25519 \
        --bootstrap scripts/bootstrap.sh

What it does, per worker (sequentially; cheap to do serially since spawn is the
slow part — actual processing is what takes hours):

  1. POST /instances/create with template=<snapshot> → instance id
  2. Poll GET /instances/list until status=RUNNING and ip is non-empty
  3. SSH (with retries) and:
       - scp bootstrap.sh + an env file to /tmp
       - source the env file + bash /tmp/bootstrap.sh
  4. Print "spawned worker N: id=… ip=…"

The workers self-terminate on empty queue (SHUTDOWN_ON_EMPTY=1), so there's no
explicit teardown step here. If something gets stuck, use `tnr delete <id>`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import paramiko
import requests


THUNDER_API = "https://api.thundercompute.com:8443"


def auth_headers() -> dict:
    token = os.environ.get("TNR_API_TOKEN") or _load_token_from_cli()
    if not token:
        sys.exit("ERROR: no Thunder API token; set TNR_API_TOKEN or run `tnr login` first")
    return {"Authorization": f"Bearer {token}"}


def _load_token_from_cli() -> str | None:
    """Read token cached by `tnr login`. CLI stores it under ~/.thunder/."""
    for cand in (
        Path.home() / ".thunder" / "cli_config.json",
        Path.home() / ".thunder" / "config.json",
        Path.home() / ".thunder" / "token",
    ):
        if cand.is_file():
            text = cand.read_text().strip()
            if cand.suffix == ".json":
                try:
                    j = json.loads(text)
                    return j.get("token") or j.get("api_token") or j.get("access_token")
                except json.JSONDecodeError:
                    continue
            return text
    return None


def add_ssh_key_to_instance(instance_id, public_key_text: str) -> None:
    """POST /instances/{id}/add_key — Thunder doesn't seed our key automatically.
    Without this, ssh fails with publickey rejected. Required after every create."""
    url = f"{THUNDER_API}/instances/{instance_id}/add_key"
    r = requests.post(url, json={"public_key": public_key_text}, headers=auth_headers(), timeout=30)
    if r.status_code >= 400:
        print(f"[spawn] WARNING: add_key returned {r.status_code} {r.text}")
    else:
        print(f"[spawn] add_key ok for instance {instance_id}")


def create_instance(args, name_suffix: int) -> dict:
    body = {
        "mode": args.mode,
        "gpu_type": args.gpu,
        "num_gpus": args.num_gpus,
        "cpu_cores": args.vcpus,
        "disk_size_gb": args.primary_disk,
        "template": args.snapshot,
    }
    if args.ephemeral_disk:
        body["ephemeral_disk_gb"] = args.ephemeral_disk
    if args.ssh_pub_key:
        body["public_key"] = Path(args.ssh_pub_key).expanduser().read_text().strip()

    print(f"[spawn] POST /instances/create body={ {k:v for k,v in body.items() if k!='public_key'} }")
    r = requests.post(f"{THUNDER_API}/instances/create", json=body, headers=auth_headers(), timeout=60)
    if r.status_code >= 400:
        print(f"[spawn] ERROR creating instance: {r.status_code} {r.text}", file=sys.stderr)
        r.raise_for_status()
    return r.json()


def wait_for_running(instance_id: str, timeout_s: int = 600) -> dict:
    """Poll /instances/list until our instance is RUNNING with an IP."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{THUNDER_API}/instances/list", headers=auth_headers(), timeout=30)
        r.raise_for_status()
        instances = r.json()
        # InstanceListResponse is map<id, InstanceListItem>
        for k, v in instances.items():
            if str(k) == str(instance_id) or v.get("uuid") == str(instance_id):
                status = (v.get("status") or "").upper()
                ip = v.get("ip") or ""
                port = v.get("port") or 22
                if status == "RUNNING" and ip:
                    return {"id": k, **v, "port": port}
                print(f"[spawn] {k}: status={status} ip={ip!r}; waiting…")
                break
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} not RUNNING after {timeout_s}s")


def ssh_connect(host: str, port: int, key_path: Path, username: str = "ubuntu",
                retries: int = 20) -> paramiko.SSHClient:
    pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))
    last_err = None
    for i in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, port=port, username=username, pkey=pkey, timeout=15, banner_timeout=15)
            return c
        except Exception as e:
            last_err = e
            print(f"[spawn] ssh {host}:{port} attempt {i+1}/{retries} failed: {e}; retry in 10s")
            time.sleep(10)
    raise RuntimeError(f"ssh to {host}:{port} failed after {retries} retries: {last_err}")


def remote_run(client: paramiko.SSHClient, cmd: str, label: str) -> int:
    chan = client.get_transport().open_session()
    chan.exec_command(cmd)
    while True:
        if chan.recv_ready():
            print(chan.recv(4096).decode(errors="replace"), end="")
        if chan.recv_stderr_ready():
            print(chan.recv_stderr(4096).decode(errors="replace"), end="", file=sys.stderr)
        if chan.exit_status_ready():
            break
        time.sleep(0.2)
    rc = chan.recv_exit_status()
    print(f"[spawn] {label} exit={rc}")
    return rc


def deploy(client: paramiko.SSHClient, bootstrap_local: Path, env_vars: dict) -> None:
    sftp = client.open_sftp()
    try:
        # Push bootstrap.sh
        remote_path = "/tmp/dynhamr_bootstrap.sh"
        sftp.put(str(bootstrap_local), remote_path)
        sftp.chmod(remote_path, 0o755)

        # Push env file (read by `source` before bootstrap)
        env_lines = "\n".join(f"export {k}={shell_quote(v)}" for k, v in env_vars.items()) + "\n"
        env_path = "/tmp/dynhamr_env.sh"
        with sftp.open(env_path, "w") as f:
            f.write(env_lines)
        sftp.chmod(env_path, 0o600)
    finally:
        sftp.close()


def shell_quote(s: str) -> str:
    if all(c.isalnum() or c in "._-/+" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def spawn_one(args, idx: int, env_vars: dict) -> dict:
    print(f"\n=== Spawning worker {idx+1}/{args.num_workers} ===")
    inst = create_instance(args, idx)
    instance_id = inst.get("identifier") or inst.get("uuid")
    print(f"[spawn] created instance id={instance_id}")

    info = wait_for_running(instance_id)
    ip = info["ip"]
    port = info.get("port") or 22
    print(f"[spawn] running: id={info['id']} ip={ip}:{port}")

    # Thunder doesn't auto-add the user's SSH key to a fresh instance; the CLI's
    # `tnr connect` calls add_key behind the scenes. We replicate that here so
    # the subsequent ssh_connect succeeds.
    pub_key_text = Path(args.ssh_pub_key).expanduser().read_text().strip()
    add_ssh_key_to_instance(info["id"], pub_key_text)
    time.sleep(3)  # small grace period for sshd to reload authorized_keys

    client = ssh_connect(ip, port, Path(args.ssh_key).expanduser())
    try:
        worker_id = f"thunder-{info['id']}-{idx:02d}"
        env_full = {**env_vars, "WORKER_ID": worker_id}
        deploy(client, Path(args.bootstrap), env_full)
        rc = remote_run(client, "set -a; source /tmp/dynhamr_env.sh; bash /tmp/dynhamr_bootstrap.sh", "bootstrap")
        if rc != 0:
            print(f"[spawn] WARNING: bootstrap rc={rc} on worker {idx} — instance left running for inspection")
    finally:
        client.close()

    return {"id": info["id"], "ip": ip, "worker_id": worker_id}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--snapshot", required=True, help="Snapshot name to template instances from")
    p.add_argument("--gpu", default="a6000", choices=["a6000", "a100", "h100"])
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--vcpus", type=int, default=8)
    p.add_argument("--mode", default="prototyping", choices=["prototyping", "production"])
    p.add_argument("--primary-disk", type=int, default=200)
    p.add_argument("--ephemeral-disk", type=int, default=0)
    p.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key matching seeded public key")
    p.add_argument("--ssh-pub-key", default="~/.ssh/id_ed25519.pub",
                   help="SSH public key to inject into the instance's authorized_keys")
    p.add_argument("--bootstrap", default="scripts/bootstrap.sh", help="Local path to bootstrap.sh")
    args = p.parse_args()

    required_env = ["DATABASE_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "PAIR_OUT_S3_PREFIX"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing env vars: {missing}")
    env_vars = {k: os.environ[k] for k in required_env}
    env_vars["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    env_vars["PIPELINE_NAME"]      = os.environ.get("PIPELINE_NAME", "handpose-v1")
    env_vars["MAX_FRAMES"]         = os.environ.get("MAX_FRAMES", "600")
    env_vars["SHUTDOWN_ON_EMPTY"]  = os.environ.get("SHUTDOWN_ON_EMPTY", "1")

    spawned = []
    for i in range(args.num_workers):
        try:
            r = spawn_one(args, i, env_vars)
            spawned.append(r)
        except Exception as e:
            print(f"[spawn] worker {i} failed: {e}", file=sys.stderr)
            # continue spawning the rest
            continue

    print(f"\n=== Done; spawned {len(spawned)}/{args.num_workers} workers ===")
    for s in spawned:
        print(f"  id={s['id']:>20}  ip={s['ip']:>16}  worker={s['worker_id']}")
    return 0 if len(spawned) == args.num_workers else 1


if __name__ == "__main__":
    sys.exit(main())
