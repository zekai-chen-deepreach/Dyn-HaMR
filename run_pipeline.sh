#!/bin/bash
# DynHAMR Full Pipeline with Rendering
# Usage: bash run_pipeline.sh <video_name_without_extension>
# Example: bash run_pipeline.sh test
#
# Input:  test/videos/<seq>.mp4
# Output: test/output/<seq>/
#           ├── <seq>_postprocessed.npz      # Final hand pose (MANO params)
#           ├── <seq>_raw.npz                # Raw optimization result
#           ├── <seq>_mesh.mp4               # Mesh overlay video (src_cam)
#           ├── images/                      # Extracted frames
#           ├── cameras/                     # VIPE camera params
#           ├── track_preds/                 # HaMeR hand detection
#           └── shot_idcs.json               # Shot segmentation

set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash run_pipeline.sh <seq_name>"
    echo "  Put your video at: test/videos/<seq_name>.mp4"
    exit 1
fi

SEQ=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/test/output/$SEQ"

source ~/miniconda3/etc/profile.d/conda.sh
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export CONDA_BASE=~/miniconda3
unset DISPLAY

echo "============================================"
echo "DynHAMR Pipeline + Render: $SEQ"
echo "============================================"

# Check input
if [ ! -f "$SCRIPT_DIR/test/videos/$SEQ.mp4" ]; then
    echo "ERROR: Video not found: test/videos/$SEQ.mp4"
    exit 1
fi

# Auto-convert to H.264 1080p (VIPE OOMs on HEVC or 4K)
CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$SCRIPT_DIR/test/videos/$SEQ.mp4")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SCRIPT_DIR/test/videos/$SEQ.mp4")
if [ "$CODEC" != "h264" ] || [ "$HEIGHT" -gt 1080 ]; then
    echo "  Converting ${CODEC} ${HEIGHT}p -> h264 1080p..."
    ffmpeg -y -i "$SCRIPT_DIR/test/videos/$SEQ.mp4" -vf "scale=-2:1080" -c:v libx264 -an "$SCRIPT_DIR/test/videos/${SEQ}_conv.mp4" 2>/dev/null
    mv "$SCRIPT_DIR/test/videos/${SEQ}_conv.mp4" "$SCRIPT_DIR/test/videos/$SEQ.mp4"
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: VIPE camera estimation
echo ""
echo "[$(date)] Step 1/5: VIPE camera estimation"
conda activate vipe
cd "$SCRIPT_DIR/third-party/vipe"
if [ -f "vipe_results/pose/$SEQ.npz" ]; then
    echo "  VIPE results already exist, skipping"
else
    vipe infer "$SCRIPT_DIR/test/videos/$SEQ.mp4" -p dynhamr
fi

# Step 2: HaMeR hand detection + data preprocessing
echo ""
echo "[$(date)] Step 2/5: HaMeR hand detection"
conda activate dynhamr
cd "$SCRIPT_DIR/dyn-hamr"
python run_opt.py data=video_vipe "data.seq=$SEQ" run_opt=False run_vis=False

# Step 3: Optimization (root_fit + smooth_fit)
echo ""
echo "[$(date)] Step 3/5: Hand pose optimization"
python run_opt.py data=video_vipe "data.seq=$SEQ" run_vis=False

# Step 4: Post-processing (vertex-space outlier fix)
echo ""
echo "[$(date)] Step 4/5: Post-processing"
NPZ=$(find "$SCRIPT_DIR/outputs/logs" -path "*${SEQ}*smooth_fit*world_results.npz" 2>/dev/null | sort | tail -1)

if [ -z "$NPZ" ]; then
    echo "ERROR: No optimization result found"
    exit 1
fi

python postprocess_npz.py --input "$NPZ" --output "$OUTPUT_DIR/${SEQ}_postprocessed.npz"

# Copy raw npz too
cp "$NPZ" "$OUTPUT_DIR/${SEQ}_raw.npz"

# Step 5: Render mesh overlay (disabled — use separate render instance)
# echo ""
# echo "[$(date)] Step 5/5: Rendering"
# LOG_DIR=$(dirname $(dirname "$NPZ"))
# cp "$OUTPUT_DIR/${SEQ}_postprocessed.npz" "$NPZ"
# python run_vis.py --log_root "$LOG_DIR" --save_root "$OUTPUT_DIR" --phases smooth_fit --render_views src_cam --overwrite
# MESH_FILE=$(ls "$OUTPUT_DIR/"*src_cam.mp4 2>/dev/null | head -1)
# [ -n "$MESH_FILE" ] && mv "$MESH_FILE" "$OUTPUT_DIR/${SEQ}_mesh.mp4"

LOG_DIR=$(dirname $(dirname "$NPZ"))

# Collect all data needed for rendering on separate instance
echo ""
echo "[$(date)] Collecting render data..."
cp -r "$SCRIPT_DIR/test/images/$SEQ" "$OUTPUT_DIR/images/" 2>/dev/null || true
cp -r "$SCRIPT_DIR/test/dynhamr/cameras/$SEQ" "$OUTPUT_DIR/cameras/" 2>/dev/null || true
cp -r "$SCRIPT_DIR/test/dynhamr/track_preds/$SEQ" "$OUTPUT_DIR/track_preds/" 2>/dev/null || true
cp "$SCRIPT_DIR/test/dynhamr/shot_idcs/$SEQ.json" "$OUTPUT_DIR/shot_idcs.json" 2>/dev/null || true
cp -r "$LOG_DIR/.hydra" "$OUTPUT_DIR/hydra_config/" 2>/dev/null || true

echo ""
echo "============================================"
echo "Done! Output: $OUTPUT_DIR/"
echo "============================================"
ls -la "$OUTPUT_DIR/"
echo ""
du -sh "$OUTPUT_DIR/"
