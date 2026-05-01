#!/usr/bin/env bash
# Bootstrap script run on every Thunder Compute worker instance after spawn.
#
# Assumes the instance was started from the dynhamr-handpose-v1 snapshot which
# already contains:
#   * /workspace/Dyn-HaMR (clone of zekai-chen-deepreach/Dyn-HaMR)
#   * Two conda envs in ~/miniconda3:
#       - `dynhamr` (torch 1.13 cu117, hamer, dyn-hamr)
#       - `vipe`    (torch 2.7 cu128, vipe + cpp extensions)
#   * /workspace/Dyn-HaMR/_DATA (HaMeR/MANO/ViTPose ckpts)
#   * /workspace/Dyn-HaMR/third-party/hamer/pretrained_models/detector.pt
#   * ~/.cache/torch/hub (DROID, grounding_dino, SAM, geocalib, ...)
#   * AWS CLI v2 installed
#
# spawn.py injects these env vars by writing /tmp/dynhamr_env.sh, which we source:
#   DATABASE_URL
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
#   PAIR_OUT_S3_PREFIX
#   (optional) WORKER_ID, PIPELINE_NAME, MAX_FRAMES, SHUTDOWN_ON_EMPTY
#
# Behaviour:
#   1. Verify env vars
#   2. git pull latest repo code (hot-fix path without rebaking snapshot)
#   3. Refresh _DATA / detector / torch_hub from S3 if missing
#      (covers the case where snapshot was taken before models were staged,
#       and is idempotent — `aws s3 sync` is a no-op when files match)
#   4. Launch worker as nohup, logged to ~/dynhamr_worker.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/Dyn-HaMR}"
LOG_FILE="${HOME}/dynhamr_worker.log"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-s3://dr-deepreach-artifacts/dynhamr}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"

echo "[bootstrap] $(date -u '+%FT%TZ') starting on $(hostname)"

# Thunder GPU AMI exposes /usr/lib/x86_64-linux-gnu/libcuda.so.1 but not the
# unsuffixed libcuda.so symlink that ultralytics/YOLO and torch's NVRTC paths
# expect. Without this, hamer YOLO inference fails with:
#   "Could not load library libcudnn_cnn_infer.so.8 — libcuda.so: cannot open"
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

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

# 1. Refresh repo to latest main (idempotent)
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] git pull in $REPO_DIR"
    git -C "$REPO_DIR" fetch --depth 1 origin main || true
    git -C "$REPO_DIR" reset --hard origin/main || true
    git -C "$REPO_DIR" submodule update --init --recursive || true
fi

# 2. Sanity check
test -x "$REPO_DIR/run_pipeline_chunked.sh" || { echo "FATAL: pipeline missing" >&2; exit 1; }
test -f "$REPO_DIR/scripts/worker.py" || { echo "FATAL: worker.py missing" >&2; exit 1; }

# 3. Sync model artifacts from S3 (idempotent — aws s3 sync skips matching files)
mkdir -p "$REPO_DIR/_DATA" \
         "$REPO_DIR/third-party/hamer/pretrained_models" \
         "$HOME/.cache/torch/hub"

echo "[bootstrap] aws s3 sync _DATA"
aws s3 sync --no-progress "$ARTIFACT_PREFIX/_DATA/" "$REPO_DIR/_DATA/" || {
    echo "[bootstrap] WARN: _DATA sync failed (perhaps already present); continuing"
}
echo "[bootstrap] aws s3 cp detector.pt"
aws s3 cp --no-progress "$ARTIFACT_PREFIX/detector.pt" \
    "$REPO_DIR/third-party/hamer/pretrained_models/detector.pt" || true
echo "[bootstrap] aws s3 sync torch_hub"
aws s3 sync --no-progress "$ARTIFACT_PREFIX/torch_hub/" "$HOME/.cache/torch/hub/" || true

# 4. Activate the worker conda env (dynhamr) and start the worker
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dynhamr

# Ensure ultralytics/YOLO finds cudnn 8 (shipped inside torch 1.13's libdir,
# not in any default loader path)
export LD_LIBRARY_PATH="$CONDA_BASE/envs/dynhamr/lib/python3.10/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Defensive: ensure worker deps (psycopg2, boto3) installed
if ! python -c "import psycopg2, boto3" 2>/dev/null; then
    pip install --no-cache-dir 'psycopg2-binary>=2.9' boto3
fi

echo "[bootstrap] launching worker → $LOG_FILE"
nohup python "$REPO_DIR/scripts/worker.py" >> "$LOG_FILE" 2>&1 &
WORKER_PID=$!
echo "[bootstrap] worker pid=$WORKER_PID"
disown $WORKER_PID || true

# Brief grace + sanity tail to surface immediate startup errors in spawn.py logs
sleep 3
echo "[bootstrap] worker tail (first lines):"
tail -n 5 "$LOG_FILE" 2>/dev/null || true
echo "[bootstrap] done"
