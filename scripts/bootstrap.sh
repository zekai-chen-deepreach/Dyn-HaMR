#!/usr/bin/env bash
# Bootstrap script run on every Thunder Compute worker instance after spawn.
#
# Assumptions:
#   - This script is baked into the snapshot at /opt/dynhamr/bootstrap.sh.
#   - The snapshot already contains:
#     * /workspace/Dyn-HaMR (clone of zekai-chen-deepreach/Dyn-HaMR)
#     * /workspace/Dyn-HaMR/_DATA (model checkpoints)
#     * conda env `dynhamr` activated by /opt/conda/etc/profile.d/conda.sh
#     * AWS CLI v2 + psycopg2 in the dynhamr env
#   - spawn.py injects these env vars via the user_data / cloud-init mechanism:
#     * DATABASE_URL
#     * AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
#     * PAIR_OUT_S3_PREFIX
#     * (optional) WORKER_ID, PIPELINE_NAME, MAX_FRAMES, SHUTDOWN_ON_EMPTY
#
# Behaviour:
#   1. git pull latest worker code (in case it was updated since snapshot was taken)
#   2. Activate conda env
#   3. nohup the worker, log to /var/log/dynhamr_worker.log
#   4. Exit (worker keeps running). If SHUTDOWN_ON_EMPTY=1 the worker will
#      `sudo shutdown -h now` after queue drains, killing the instance.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/Dyn-HaMR}"
LOG_FILE="/var/log/dynhamr_worker.log"

echo "[bootstrap] $(date -u '+%FT%TZ') starting on $(hostname)"

# Verify env vars
: "${DATABASE_URL:?DATABASE_URL not set}"
: "${PAIR_OUT_S3_PREFIX:?PAIR_OUT_S3_PREFIX not set}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID not set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY not set}"
: "${AWS_DEFAULT_REGION:=us-west-2}"

export AWS_DEFAULT_REGION
export PIPELINE_NAME="${PIPELINE_NAME:-handpose-v1}"
export WORKER_ID="${WORKER_ID:-$(hostname)}"
export SHUTDOWN_ON_EMPTY="${SHUTDOWN_ON_EMPTY:-1}"
export MAX_FRAMES="${MAX_FRAMES:-600}"
export REPO_DIR

# 1. Refresh repo (no-op if upstream hasn't changed; allows hot-fixing worker.py
# without rebaking the snapshot)
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] git pull in $REPO_DIR"
    git -C "$REPO_DIR" fetch --depth 1 origin main || true
    git -C "$REPO_DIR" reset --hard origin/main || true
    git -C "$REPO_DIR" submodule update --init --recursive || true
fi

# 2. Sanity: pipeline + worker exist
test -x "$REPO_DIR/run_pipeline_chunked.sh" || {
    echo "[bootstrap] FATAL: $REPO_DIR/run_pipeline_chunked.sh missing/non-executable" >&2
    exit 1
}
test -f "$REPO_DIR/scripts/worker.py" || {
    echo "[bootstrap] FATAL: $REPO_DIR/scripts/worker.py missing" >&2
    exit 1
}

# 3. Activate conda env, install psycopg2 if missing (idempotent)
source /opt/conda/etc/profile.d/conda.sh
conda activate dynhamr
if ! python -c "import psycopg2" 2>/dev/null; then
    pip install --no-cache-dir 'psycopg2-binary>=2.9'
fi

# 4. Launch worker, detached, logging
echo "[bootstrap] launching worker → $LOG_FILE"
sudo touch "$LOG_FILE" && sudo chown "$USER" "$LOG_FILE" || touch "$LOG_FILE"
nohup python "$REPO_DIR/scripts/worker.py" >> "$LOG_FILE" 2>&1 &
WORKER_PID=$!
echo "[bootstrap] worker pid=$WORKER_PID"
disown $WORKER_PID || true

echo "[bootstrap] done; worker running in background"
