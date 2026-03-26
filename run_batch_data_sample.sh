#!/bin/bash
# Batch process all Data Sample videos through DynHAMR + upperbody pipeline
# Usage: bash run_batch_data_sample.sh

set -e
cd /home/deepreach/Zekai_code/Dyn-HaMR

SEQS=(
    car-cleaning-episode-1
    car-cleaning-episode-2
    car-cleaning-episode-3
    electrician-episode-1
    electrician-episode-2
    electrician-episode-3
    electrician-episode-4
    electrician-episode-5
    electrician-episode-6
    electrician-episode-7
    electrician-episode-8
    electrician-episode-9
    electrician-episode-10
    electrician-episode-11
    electrician-episode-12
    electrician-episode-13
    gas-station-refuel-episode-1
    gas-station-refuel-episode-2
    office-cleaning-episode-1
    office-cleaning-episode-2
    office-cleaning-episode-3
    office-cleaning-episode-4
)

DYNHAMR_OUT_BASE="/home/deepreach/Zekai_code/asi-human-data/thirdparty/dynhamr/outputs/logs/video-custom"
UPPERBODY_DIR="/home/deepreach/Zekai_code/asi-human-data/upperbody"
RESULTS_LOG="/tmp/batch_results.txt"
> "$RESULTS_LOG"

echo "=== Batch processing ${#SEQS[@]} videos ==="
echo ""

for seq in "${SEQS[@]}"; do
    echo "============================================"
    echo "Processing: $seq"
    echo "============================================"

    # Step 1: Run DynHAMR (frame extraction + HaMeR + VIPE + root_fit + smooth_fit)
    echo "[1/4] Running DynHAMR..."
    conda run -n dynhamr python dyn-hamr/run_opt.py \
        data=video_vipe data.seq="$seq" run_prior=False 2>&1 | tail -5

    # Find the output NPZ
    DATE_DIR=$(ls -td "$DYNHAMR_OUT_BASE"/*/  2>/dev/null | head -1)
    SEQ_DIR=$(find "$DYNHAMR_OUT_BASE" -maxdepth 2 -type d -name "${seq}-*" | sort -r | head -1)
    if [ -z "$SEQ_DIR" ]; then
        echo "ERROR: No output found for $seq, skipping"
        echo "$seq: FAILED (no DynHAMR output)" >> "$RESULTS_LOG"
        continue
    fi

    NPZ=$(find "$SEQ_DIR/smooth_fit" -name "*_world_results.npz" | head -1)
    if [ -z "$NPZ" ]; then
        echo "ERROR: No smooth_fit NPZ for $seq, skipping"
        echo "$seq: FAILED (no smooth_fit NPZ)" >> "$RESULTS_LOG"
        continue
    fi
    echo "  DynHAMR NPZ: $NPZ"

    # Step 2: Generate L/R arm masks with SAM3
    FRAMES_DIR="test/images/$seq"
    LR_MASKS="/tmp/${seq}_lr_arms.npz"
    if [ ! -f "$LR_MASKS" ]; then
        echo "[2/4] Generating SAM3 L/R arm masks..."
        conda run -n sam3 python upperbody/generate_lr_arm_masks.py \
            --frames-dir "$FRAMES_DIR" \
            --output "$LR_MASKS" \
            --resize 0.25 2>&1 | tail -5
    else
        echo "[2/4] SAM3 masks already exist: $LR_MASKS"
    fi

    # Step 3: Run upperbody IK
    echo "[3/4] Running upperbody IK..."
    cd /home/deepreach/Zekai_code/asi-human-data
    conda run -n dynhamr python upperbody/run_dynhamr_npz_upperbody.py \
        "$NPZ" --height 1.7 --fps 29.97 --batch-size 24 \
        --lr-arm-masks "$LR_MASKS" 2>&1 | tail -3
    cd /home/deepreach/Zekai_code/Dyn-HaMR

    # Find upperbody JSON
    UB_JSON=$(find "$SEQ_DIR/smooth_fit" -name "upperbody__*.json" ! -name "*_meta.json" | head -1)

    # Step 4: Visualize
    echo "[4/4] Generating visualization..."
    cd /home/deepreach/Zekai_code/asi-human-data
    conda run -n dynhamr python upperbody/visualize_dynhamr_upperbody.py \
        "$UB_JSON" --preview \
        --npz-path "$NPZ" \
        --frames-dir "/home/deepreach/Zekai_code/Dyn-HaMR/$FRAMES_DIR" \
        --fps 29.97 2>&1 | tail -2
    cd /home/deepreach/Zekai_code/Dyn-HaMR

    UB_VIDEO=$(find "$SEQ_DIR/smooth_fit" -name "*_preview.mp4" | head -1)

    echo ""
    echo "  DynHAMR NPZ:      $NPZ"
    echo "  Upperbody video:   $UB_VIDEO"
    echo "$seq: $NPZ | $UB_VIDEO" >> "$RESULTS_LOG"
    echo ""
done

echo ""
echo "============================================"
echo "ALL DONE. Results summary:"
echo "============================================"
cat "$RESULTS_LOG"
