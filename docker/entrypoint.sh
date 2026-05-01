#!/usr/bin/env bash
# Dyn-HaMR pair generator entrypoint for AWS Batch.
#
# Required env vars (passed by Batch via job parameters):
#   VIDEO_S3            full s3:// URI of one source video (≤ ~5 min recommended)
#   PAIR_OUT_S3         s3:// prefix where pair NPZ + per-task manifest go
#                         e.g. s3://your-bucket/pair_dataset/
#   PAIR_GLOBAL_START   integer; the global pair index this video's first chunk takes
#                         (chunks within this video → PAIR_GLOBAL_START, +1, +2, …)
#
# Optional env vars:
#   MAX_FRAMES          per-chunk max frames (default 600 ≈ 20s at 30fps)
#   AWS_BATCH_JOB_ID    used to namespace per-task manifest filename (auto-set by Batch)
#   PIPELINE_EXTRA      extra flags forwarded to run_pipeline_chunked.sh
#
# Behaviour:
#   1. Pull source video from S3 to local scratch
#   2. Run run_pipeline_chunked.sh with --no-render --emit-pairs <local>
#   3. Sync pair NPZs and per-task manifest to S3
#   4. Exit 0 on success; non-zero is treated as job failure by Batch (will be retried per job def)
#
set -euo pipefail

: "${VIDEO_S3:?VIDEO_S3required}"
: "${PAIR_OUT_S3:?PAIR_OUT_S3 required}"
: "${PAIR_GLOBAL_START:?PAIR_GLOBAL_START required}"

MAX_FRAMES="${MAX_FRAMES:-600}"
JOB_ID="${AWS_BATCH_JOB_ID:-local-$(date +%s)-$$}"
EXTRA="${PIPELINE_EXTRA:-}"

WORK="${SCRATCH_DIR:-/tmp/dynhamr_job}"
INPUT_LOCAL="$WORK/input.mp4"
OUTPUT_LOCAL="$WORK/output"
PAIRS_LOCAL="$WORK/pairs"
mkdir -p "$WORK" "$OUTPUT_LOCAL" "$PAIRS_LOCAL"

# Per-task manifest path (run_pipeline appends to manifest.jsonl in pairs root)
echo "[entrypoint] job_id=$JOB_ID"
echo "[entrypoint] VIDEO_S3=$VIDEO_S3"
echo "[entrypoint] PAIR_OUT_S3=$PAIR_OUT_S3"
echo "[entrypoint] PAIR_GLOBAL_START=$PAIR_GLOBAL_START"
echo "[entrypoint] MAX_FRAMES=$MAX_FRAMES"

# 1. Download
echo "[entrypoint] Downloading source video..."
aws s3 cp "$VIDEO_S3" "$INPUT_LOCAL"
ls -la "$INPUT_LOCAL"

# 2. Activate env and run the pipeline
source /opt/conda/etc/profile.d/conda.sh
conda activate dynhamr

cd /workspace
set -x
bash run_pipeline_chunked.sh \
    "$INPUT_LOCAL" \
    "$OUTPUT_LOCAL" \
    --no-render \
    --max-frames "$MAX_FRAMES" \
    --emit-pairs "$PAIRS_LOCAL" \
    --pair-global-start "$PAIR_GLOBAL_START" \
    $EXTRA
set +x

# 3. Rename per-task manifest to avoid S3 sync collisions across concurrent jobs.
#    Each job owns its own manifest file; an aggregator merges them later.
if [ -f "$PAIRS_LOCAL/manifest.jsonl" ]; then
    mv "$PAIRS_LOCAL/manifest.jsonl" "$PAIRS_LOCAL/manifests_${JOB_ID//\//_}.jsonl"
    mkdir -p "$PAIRS_LOCAL/manifests"
    mv "$PAIRS_LOCAL/manifests_${JOB_ID//\//_}.jsonl" "$PAIRS_LOCAL/manifests/manifests_${JOB_ID//\//_}.jsonl"
fi

# 4. Upload pair NPZs (shards/) and the renamed manifest
echo "[entrypoint] Uploading pairs to $PAIR_OUT_S3 ..."
aws s3 sync --no-progress "$PAIRS_LOCAL/" "$PAIR_OUT_S3" --exclude "*" --include "shards/*" --include "manifests/*"

# 5. Free scratch (optional; Batch reclaims the container regardless)
rm -rf "$WORK"

echo "[entrypoint] Done"
