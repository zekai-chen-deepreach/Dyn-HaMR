#!/bin/bash
# DynHAMR Full Pipeline: Video -> NPZ (hand pose)
# Usage: bash run_pipeline.sh <video_name_without_extension>
# Example: bash run_pipeline.sh test
#
# Input:  test/videos/<seq>.mp4
# Output: test/<seq>_postprocessed.npz
#         outputs/logs/video-custom/<date>/<seq>-all-shot-0-0--1/smooth_fit/<seq>_000060_world_results.npz

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash run_pipeline.sh <seq_name>"
    echo "  Put your video at: test/videos/<seq_name>.mp4"
    exit 1
fi

SEQ=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export CONDA_BASE=~/miniconda3
unset DISPLAY

echo "============================================"
echo "DynHAMR Pipeline: $SEQ"
echo "============================================"

# Check input
if [ ! -f "$SCRIPT_DIR/test/videos/$SEQ.mp4" ]; then
    echo "ERROR: Video not found: test/videos/$SEQ.mp4"
    exit 1
fi

# Step 1: VIPE camera estimation
echo ""
echo "[$(date)] Step 1/4: VIPE camera estimation"
conda activate vipe
cd "$SCRIPT_DIR/third-party/vipe"
if [ -f "vipe_results/pose/$SEQ.npz" ]; then
    echo "  VIPE results already exist, skipping"
else
    vipe infer "$SCRIPT_DIR/test/videos/$SEQ.mp4" -p dynhamr
fi

# Step 2: HaMeR hand detection + data preprocessing
echo ""
echo "[$(date)] Step 2/4: HaMeR hand detection"
conda activate dynhamr
cd "$SCRIPT_DIR/dyn-hamr"
python run_opt.py data=video_vipe "data.seq=$SEQ" run_opt=False run_vis=False

# Step 3: Optimization (root_fit + smooth_fit)
echo ""
echo "[$(date)] Step 3/4: Hand pose optimization"
python run_opt.py data=video_vipe "data.seq=$SEQ" run_vis=False

# Step 4: Post-processing (vertex-space outlier fix)
echo ""
echo "[$(date)] Step 4/4: Post-processing"
NPZ=$(find "$SCRIPT_DIR/outputs/logs" -path "*${SEQ}*smooth_fit*world_results.npz" 2>/dev/null | sort | tail -1)

if [ -z "$NPZ" ]; then
    echo "ERROR: No optimization result found"
    exit 1
fi

python postprocess_npz.py --input "$NPZ" --output "$SCRIPT_DIR/test/${SEQ}_postprocessed.npz"

echo ""
echo "============================================"
echo "Done! Results:"
echo "  Raw NPZ:         $NPZ"
echo "  Postprocessed:   $SCRIPT_DIR/test/${SEQ}_postprocessed.npz"
echo "============================================"
