#!/bin/bash
# Test script: run full optimization + visualization for head-6
# with the lean output changes, measuring timing for each stage.
set -e

SEQ="multi-cameras-head-6"
PROJECT_DIR="/home/deepreach/Zekai_code/Dyn-HaMR"
WORK_DIR="$PROJECT_DIR/dyn-hamr"
TODAY=$(date +%Y-%m-%d)

echo "=============================================="
echo " Dyn-HaMR Lean Output Test — $SEQ"
echo " $(date)"
echo "=============================================="

# Verify prerequisites exist
echo ""
echo "[Check] HaMeR results..."
TRACK_DIR="$PROJECT_DIR/test/dynhamr/track_preds/$SEQ"
TRACK_COUNT=$(ls "$TRACK_DIR" 2>/dev/null | wc -l)
echo "  Found $TRACK_COUNT track prediction files"

echo "[Check] VIPE results..."
VIPE_POSE="$PROJECT_DIR/third-party/vipe/vipe_results/pose/$SEQ.npz"
if [ -f "$VIPE_POSE" ]; then
    echo "  Found $VIPE_POSE"
else
    echo "  ERROR: VIPE pose not found at $VIPE_POSE"
    exit 1
fi

echo "[Check] Extracted frames..."
FRAME_DIR="$PROJECT_DIR/test/images/$SEQ"
FRAME_COUNT=$(ls "$FRAME_DIR"/*.jpg 2>/dev/null | wc -l)
echo "  Found $FRAME_COUNT frames"

# Expected output directory
OUT_DIR="$PROJECT_DIR/outputs/logs/video-custom/$TODAY/${SEQ}-all-shot-0-0--1"
echo ""
echo "[Info] Output will go to: $OUT_DIR"

# Clean if already exists (re-run today)
if [ -d "$OUT_DIR" ]; then
    echo "[Clean] Removing existing output dir for today..."
    rm -rf "$OUT_DIR"
fi

echo ""
echo "=============================================="
echo " Stage 3a: Optimization (root_fit + smooth_fit)"
echo "=============================================="
OPT_START=$(date +%s)

cd "$WORK_DIR"
conda run -n dynhamr python run_opt.py \
    data=video_vipe \
    data.seq=$SEQ \
    run_opt=True \
    run_vis=False

OPT_END=$(date +%s)
OPT_TIME=$((OPT_END - OPT_START))
echo ""
echo ">>> Optimization took: ${OPT_TIME}s ($(echo "scale=1; $OPT_TIME/60" | bc)m)"

echo ""
echo "=============================================="
echo " Stage 3b: Visualization (grid video)"
echo "=============================================="
VIS_START=$(date +%s)

cd "$WORK_DIR"
conda run -n dynhamr python run_opt.py \
    data=video_vipe \
    data.seq=$SEQ \
    run_opt=False \
    run_vis=True

VIS_END=$(date +%s)
VIS_TIME=$((VIS_END - VIS_START))
echo ""
echo ">>> Visualization took: ${VIS_TIME}s ($(echo "scale=1; $VIS_TIME/60" | bc)m)"

TOTAL_TIME=$((VIS_END - OPT_START))
echo ""
echo "=============================================="
echo " Results Summary"
echo "=============================================="
echo "Optimization: ${OPT_TIME}s"
echo "Visualization: ${VIS_TIME}s"
echo "Total:         ${TOTAL_TIME}s ($(echo "scale=1; $TOTAL_TIME/60" | bc)m)"
echo ""

echo "--- Output files ---"
if [ -d "$OUT_DIR" ]; then
    echo "Directory: $OUT_DIR"
    echo ""
    # List all files with sizes
    find "$OUT_DIR" -type f -printf '%s\t%p\n' | sort -k2 | \
        awk '{
            size=$1;
            path=$2;
            if (size > 1048576) printf "%7.1fM  %s\n", size/1048576, path;
            else if (size > 1024) printf "%7.1fK  %s\n", size/1024, path;
            else printf "%7dB  %s\n", size, path;
        }'
    echo ""
    # Count files by type
    echo "--- File count by extension ---"
    find "$OUT_DIR" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn
    echo ""
    # Total size
    TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)
    echo "Total output size: $TOTAL_SIZE"
else
    echo "ERROR: Output directory not found!"
fi

echo ""
echo "Done! $(date)"
