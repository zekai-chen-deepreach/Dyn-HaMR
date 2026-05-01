"""
DynHAMR pair-data worker.

Runs on a Thunder Compute A6000 instance. Polls the dr_segment_processing
work queue (Postgres FOR UPDATE SKIP LOCKED), pulls one segment at a time
from S3, runs run_pipeline_chunked.sh in pair-emit mode, uploads the resulting
pair NPZ + manifest jsonl to S3, then marks done. Exits and shuts the box
down when the queue is empty for too long.

Env vars (required):
  DATABASE_URL          postgres://… connection string for prod RDS
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION    e.g. us-west-2
  PAIR_OUT_S3_PREFIX    e.g. s3://dr-handpose-outputs/v1/ (trailing slash)

Env vars (optional, with defaults):
  PIPELINE_NAME         dr_segment_processing.pipeline filter (default: handpose-v1)
  WORKER_ID             override machine identifier (default: socket.gethostname())
  SHUTDOWN_ON_EMPTY     '1' to `sudo shutdown -h now` once queue is empty (default '0')
  EMPTY_POLLS_BEFORE_EXIT  consecutive empty polls before exiting (default: 5)
  EMPTY_POLL_SLEEP_SEC  sleep between empty polls (default: 30)
  WORK_DIR              local scratch root (default: /tmp/dynhamr_worker)
  REPO_DIR              path to the cloned Dyn-HaMR repo (default: /workspace/Dyn-HaMR)
  MAX_FRAMES            per-chunk max frames forwarded to pipeline (default: 600)

Notes:
  - The pipeline command currently lives at $REPO_DIR/run_pipeline_chunked.sh.
  - One worker = one GPU. Concurrency is across machines, not within a machine.
  - The claim transaction is short (claim → commit). Heavy work happens outside
    the txn so we don't hold row locks during VIPE/HaMeR/optim (4-6 min/chunk).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extensions


# ── Config ─────────────────────────────────────────────────────────────────

DB_URL          = os.environ["DATABASE_URL"]
AWS_REGION      = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PAIR_OUT_PREFIX = os.environ["PAIR_OUT_S3_PREFIX"].rstrip("/") + "/"
PIPELINE_NAME   = os.environ.get("PIPELINE_NAME", "handpose-v1")
WORKER_ID       = os.environ.get("WORKER_ID", socket.gethostname())
SHUTDOWN_ON_EMPTY    = os.environ.get("SHUTDOWN_ON_EMPTY", "0") == "1"
EMPTY_POLLS_BEFORE_EXIT = int(os.environ.get("EMPTY_POLLS_BEFORE_EXIT", "5"))
EMPTY_POLL_SLEEP_SEC = int(os.environ.get("EMPTY_POLL_SLEEP_SEC", "30"))
WORK_DIR        = Path(os.environ.get("WORK_DIR", "/tmp/dynhamr_worker"))
REPO_DIR        = Path(os.environ.get("REPO_DIR", "/workspace/Dyn-HaMR"))
MAX_FRAMES      = int(os.environ.get("MAX_FRAMES", "600"))

PIPELINE_SH     = REPO_DIR / "run_pipeline_chunked.sh"
LOG_PREFIX      = f"[worker {WORKER_ID}]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# ── DB helpers ─────────────────────────────────────────────────────────────

def db_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn


def claim_one(conn) -> Optional[dict]:
    """Atomically claim one pending row. Returns None if the queue is empty."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, "filterJobId", "segmentFile"
            FROM dr_segment_processing
            WHERE pipeline = %s AND status = 'downloaded'
            ORDER BY id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        """, (PIPELINE_NAME,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return None
        row_id, filter_job_id, segment_file = row
        cur.execute("""
            UPDATE dr_segment_processing
            SET status = 'processing',
                "updatedAt" = now(),
                notes = COALESCE(notes, '') || ' | claimed_by=' || %s || ' at ' || now()::text
            WHERE id = %s
        """, (WORKER_ID, row_id))
        conn.commit()
        return {"id": row_id, "filter_job_id": filter_job_id, "segment_file": segment_file}


def build_s3_input_uri(conn, filter_job_id: str, segment_file: str) -> str:
    """Resolve segmentsBucket + segmentsPrefix from dr_vendor_filter_job → full s3 uri."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT "segmentsBucket", "segmentsPrefix"
            FROM dr_vendor_filter_job
            WHERE id = %s
        """, (filter_job_id,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"filter_job {filter_job_id} not found")
        bucket, prefix = row
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return f"s3://{bucket}/{prefix or ''}{segment_file}"


def mark_done(conn, row_id: str, n_pairs: int, output_s3_uri: str) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE dr_segment_processing
            SET status = 'done',
                "updatedAt" = now(),
                notes = COALESCE(notes, '') || ' | done by ' || %s
                        || ' pairs=' || %s
                        || ' output=' || %s
                        || ' at ' || now()::text
            WHERE id = %s
        """, (WORKER_ID, n_pairs, output_s3_uri, row_id))
    conn.commit()


def mark_failed(conn, row_id: str, exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE dr_segment_processing
            SET status = 'failed',
                "updatedAt" = now(),
                notes = COALESCE(notes, '') || ' | failed by ' || %s
                        || ' at ' || now()::text
                        || ' err=' || %s
            WHERE id = %s
        """, (WORKER_ID, tb, row_id))
    conn.commit()


# ── Filesystem / subprocess helpers ───────────────────────────────────────

def aws_cp(src: str, dst: str, recursive: bool = False) -> None:
    cmd = ["aws", "s3", "cp", src, dst]
    if recursive:
        cmd.append("--recursive")
    log(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@contextmanager
def scratch_dir(name: str):
    """Per-segment scratch under WORK_DIR, removed on exit (success or failure)."""
    d = WORK_DIR / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── Per-segment work ──────────────────────────────────────────────────────

def process_segment(conn, claim: dict) -> tuple[int, str]:
    """
    Returns (n_pairs_emitted, s3_output_prefix).
    """
    seg_file = claim["segment_file"]
    seg_stem = Path(seg_file).stem  # e.g. '00001' or 'abc-clip-12'

    s3_in_uri  = build_s3_input_uri(conn, claim["filter_job_id"], seg_file)
    s3_out_uri = PAIR_OUT_PREFIX  # global namespace; pair_writer shards by global idx

    with scratch_dir(seg_stem) as work:
        local_in  = work / seg_file
        local_pairs = work / "pairs"
        local_pairs.mkdir(parents=True, exist_ok=True)

        # 1. Download
        aws_cp(s3_in_uri, str(local_in))

        # 2. Run pipeline. We pass --pair-global-start derived from the segment
        # stem when it's an integer (00001 → 1 → start_idx 0; 1*MAX_CHUNKS_PER_SEG=10 → 10).
        # If the stem isn't numeric, fall back to a hash bucket so different
        # segments don't collide on the same shard slot.
        try:
            pair_global_start = int(seg_stem) * 100  # 100 pairs reserved per segment
        except ValueError:
            # Stable hash bucket for non-numeric stems.
            pair_global_start = (abs(hash(seg_stem)) % 1_000_000) * 100

        log(f"running pipeline on {seg_file} (pair_global_start={pair_global_start})")
        cmd = [
            "bash", str(PIPELINE_SH),
            str(local_in),
            str(work / "pipeline_out"),
            "--no-render",
            "--max-frames", str(MAX_FRAMES),
            "--emit-pairs", str(local_pairs),
            "--pair-global-start", str(pair_global_start),
        ]
        log(f"$ {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        log(f"pipeline done in {time.time()-t0:.0f}s")

        # 3. Rename per-segment manifest to avoid concurrent-writer collisions
        #    when many workers all sync into the same prefix.
        manifest_src = local_pairs / "manifest.jsonl"
        if manifest_src.exists():
            manifests_dir = local_pairs / "manifests"
            manifests_dir.mkdir(exist_ok=True)
            mf_dst = manifests_dir / f"manifest_{seg_stem}_{uuid.uuid4().hex[:8]}.jsonl"
            shutil.move(str(manifest_src), str(mf_dst))

        # 4. Count pairs for the DB row
        n_pairs = sum(1 for _ in local_pairs.glob("shards/*/*.npz"))
        log(f"emitted {n_pairs} pairs for segment {seg_stem}")

        # 5. Upload everything under local_pairs/ to S3 prefix
        aws_cp(str(local_pairs) + "/", s3_out_uri, recursive=True)

    return n_pairs, s3_out_uri


# ── Main loop ─────────────────────────────────────────────────────────────

def main() -> int:
    log(f"starting worker; pipeline={PIPELINE_NAME} pair_out={PAIR_OUT_PREFIX}")
    if not PIPELINE_SH.exists():
        log(f"ERROR: pipeline script not found at {PIPELINE_SH}")
        return 2

    conn = db_connect()
    empty_streak = 0

    while True:
        try:
            claim = claim_one(conn)
        except Exception as e:
            log(f"DB error during claim: {e}; reconnecting in 30s")
            time.sleep(30)
            try:
                conn.close()
            except Exception:
                pass
            conn = db_connect()
            continue

        if claim is None:
            empty_streak += 1
            log(f"queue empty (streak {empty_streak}/{EMPTY_POLLS_BEFORE_EXIT})")
            if empty_streak >= EMPTY_POLLS_BEFORE_EXIT:
                log("queue persistently empty; exiting main loop")
                break
            time.sleep(EMPTY_POLL_SLEEP_SEC)
            continue

        empty_streak = 0
        seg_file = claim["segment_file"]
        log(f"claimed row id={claim['id']} segment={seg_file}")

        try:
            n_pairs, s3_out = process_segment(conn, claim)
            mark_done(conn, claim["id"], n_pairs, s3_out)
            log(f"marked done: id={claim['id']} pairs={n_pairs}")
        except Exception as e:
            log(f"FAILED segment {seg_file}: {e}")
            try:
                mark_failed(conn, claim["id"], e)
            except Exception as e2:
                log(f"also failed to mark_failed: {e2}")

    try:
        conn.close()
    except Exception:
        pass

    if SHUTDOWN_ON_EMPTY:
        log("SHUTDOWN_ON_EMPTY=1 → sudo shutdown -h now")
        # Best-effort; if worker doesn't have sudo, this just logs and exits.
        try:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
        except Exception as e:
            log(f"shutdown failed: {e}")

    log("worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
