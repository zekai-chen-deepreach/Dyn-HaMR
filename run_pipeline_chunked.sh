#!/bin/bash
# DynHAMR Pipeline with Auto Chunking — supports long videos
#
# Splits videos > MAX_FRAMES into chunks, processes each, then merges results.
# Output is a single unified NPZ + mesh video, equivalent to processing the
# full video in one pass.
#
# Usage:
#   bash run_pipeline_chunked.sh <video_path> [output_dir] [--no-render] [--max-frames N]
#
# Example:
#   bash run_pipeline_chunked.sh ~/videos/long_clip.mp4
#   bash run_pipeline_chunked.sh ~/videos/long_clip.mp4 ~/output/clip --max-frames 600
#
# Pipeline:
#   1. Convert to H.264 1080p (auto downscale if 4K)
#   2. Split into ≤MAX_FRAMES chunks if video is longer
#   3. Per chunk: VIPE + HaMeR + smooth_fit optimization
#   4. Postprocess each chunk (SLERP outlier fix)
#   5. Merge chunks (camera + hand alignment at boundaries)
#   6. Render mesh video (optional, full sequence)
#   7. Cleanup intermediates

set -eo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash run_pipeline_chunked.sh <video_path> [output_dir] [--no-render] [--max-frames N]"
    exit 1
fi

INPUT_VIDEO="$(realpath "$1")"
shift

OUTPUT_DIR=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_DIR="$(realpath "$1")"
    shift
fi

# Default: <video_dir>/<seq>_dynhamr_output/
if [[ -z "$OUTPUT_DIR" ]]; then
    SEQ_DEFAULT="$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')"
    OUTPUT_DIR="$(dirname "$INPUT_VIDEO")/${SEQ_DEFAULT}_dynhamr_output"
fi

DO_RENDER=true
MAX_FRAMES=600
EMIT_PAIRS_ROOT=""
PAIR_GLOBAL_START=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-render) DO_RENDER=false; shift ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --emit-pairs) EMIT_PAIRS_ROOT="$(realpath "$2")"; shift 2 ;;
        --pair-global-start) PAIR_GLOBAL_START="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: $INPUT_VIDEO not found"; exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNHAMR_DIR="$SCRIPT_DIR/dyn-hamr"
VIPE_DIR="$SCRIPT_DIR/third-party/vipe"
TEST_DIR="$SCRIPT_DIR/test"
VIDEO_DIR="$TEST_DIR/videos"
LOG_ROOT="${DYNHAMR_LOG_ROOT:-$SCRIPT_DIR/outputs/logs}"

# Export hydra config knobs so run_opt.py subprocess sees consistent paths
export DYNHAMR_TEST_ROOT="$TEST_DIR"
export DYNHAMR_VIPE_DIR="$VIPE_DIR/vipe_results"

source ~/miniconda3/etc/profile.d/conda.sh
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
unset DISPLAY
conda activate dynhamr

# Make ultralytics/YOLO + torch's CUDA loader find cudnn 8 + libcuda.so on
# Thunder/cloud AMIs that don't expose them via default loader paths.
TORCH_LIB="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
[ -n "$TORCH_LIB" ] && export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

PYTHON="$(which python)"
# Prefer the env-local binary (faster, isolated), fall back to system PATH.
FFMPEG="$HOME/miniconda3/envs/dynhamr/bin/ffmpeg"
[ ! -x "$FFMPEG" ] && FFMPEG="$(command -v ffmpeg)"
FFPROBE="$HOME/miniconda3/envs/dynhamr/bin/ffprobe"
[ ! -x "$FFPROBE" ] && FFPROBE="$(command -v ffprobe)"
[ -z "$FFMPEG" ]  && { echo "ERROR: ffmpeg not found";  exit 1; }
[ -z "$FFPROBE" ] && { echo "ERROR: ffprobe not found"; exit 1; }

mkdir -p "$VIDEO_DIR" "$OUTPUT_DIR"

# Sanitize sequence name (lowercase, replace non-alnum with -)
RAW_SEQ="$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')"
SEQ="$(echo "$RAW_SEQ" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')"

echo "============================================"
echo "DynHAMR Chunked Pipeline"
echo "  Input:  $INPUT_VIDEO"
echo "  Output: $OUTPUT_DIR"
echo "  SEQ:    $SEQ"
echo "  MAX_FRAMES per chunk: $MAX_FRAMES"
echo "  Render: $DO_RENDER"
echo "  Emit pairs: ${EMIT_PAIRS_ROOT:-no}"
[ -n "$EMIT_PAIRS_ROOT" ] && echo "  Pair global start: $PAIR_GLOBAL_START"
echo "============================================"

# Capture all stdout/stderr to a log
exec > >(tee "$OUTPUT_DIR/pipeline.log") 2>&1

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Convert + downscale to H.264 1080p
# ──────────────────────────────────────────────────────────────────────────────
SRC_VIDEO="$VIDEO_DIR/${SEQ}.mp4"
echo ""
echo "[$(date '+%H:%M:%S')] Step 1: Convert to H.264 1080p"

CODEC=$($FFPROBE -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$INPUT_VIDEO")
HEIGHT=$($FFPROBE -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT_VIDEO")

if [ "$CODEC" != "h264" ] || [ "$HEIGHT" -gt 1080 ]; then
    echo "  Converting ${CODEC} ${HEIGHT}p -> h264 1080p..."
    $FFMPEG -y -i "$INPUT_VIDEO" -vf "scale=-2:'min(1080,ih)'" -c:v libx264 -crf 18 -preset medium -an "$SRC_VIDEO" </dev/null 2>&1 | tail -3
else
    cp "$INPUT_VIDEO" "$SRC_VIDEO"
fi

# Get final dims after conversion
NUM_FRAMES=$($FFPROBE -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$SRC_VIDEO")
FPS_FRAC=$($FFPROBE -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$SRC_VIDEO")
FPS_NUM="${FPS_FRAC%/*}"; FPS_DEN="${FPS_FRAC#*/}"
FPS=$(( (FPS_NUM + FPS_DEN/2) / FPS_DEN ))
WIDTH=$($FFPROBE -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$SRC_VIDEO")
HEIGHT=$($FFPROBE -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SRC_VIDEO")
echo "  Resolved: ${WIDTH}x${HEIGHT}, ${FPS}fps, ${NUM_FRAMES} frames"

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Split into chunks if needed
# ──────────────────────────────────────────────────────────────────────────────
SEQUENCES=()
echo ""
echo "[$(date '+%H:%M:%S')] Step 2: Chunk split"

if (( NUM_FRAMES > MAX_FRAMES )); then
    N_PARTS=$(( (NUM_FRAMES + MAX_FRAMES - 1) / MAX_FRAMES ))
    PART_DURATION=$(echo "scale=4; $MAX_FRAMES / $FPS" | bc)
    echo "  $NUM_FRAMES frames > $MAX_FRAMES, splitting into $N_PARTS parts"

    for (( i=0; i<N_PARTS; i++ )); do
        PART_SEQ="${SEQ}-p$((i+1))"
        PART_VIDEO="$VIDEO_DIR/${PART_SEQ}.mp4"
        START_SEC=$(echo "scale=4; $i * $PART_DURATION" | bc)
        echo "  Part $((i+1))/$N_PARTS: start=${START_SEC}s, max ${MAX_FRAMES} frames"
        $FFMPEG -y -ss "$START_SEC" -i "$SRC_VIDEO" -frames:v "$MAX_FRAMES" \
            -c:v libx264 -crf 18 -preset medium -an "$PART_VIDEO" </dev/null 2>&1 | tail -1
        SEQUENCES+=("$PART_SEQ")
    done
else
    echo "  $NUM_FRAMES <= $MAX_FRAMES, no split"
    SEQUENCES+=("$SEQ")
fi

echo "  Sequences to process: ${SEQUENCES[*]}"

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Run DynHAMR per chunk
# ──────────────────────────────────────────────────────────────────────────────
cd "$DYNHAMR_DIR"

clear_caches() {
    local seq=$1
    rm -f "$VIPE_DIR/vipe_results/pose/${seq}.npz"
    rm -f "$VIPE_DIR/vipe_results/intrinsics/${seq}.npz"
    rm -rf "$TEST_DIR/dynhamr/track_preds/${seq}"
    rm -rf "$TEST_DIR/images/${seq}"
    rm -rf "$TEST_DIR/dynhamr/cameras/${seq}"
    rm -rf "$TEST_DIR/dynhamr/shot_idcs/${seq}.json"
    rm -rf "$TEST_DIR/dynhamr/hamer_out/${seq}"
    find "$LOG_ROOT" -maxdepth 3 -type d -name "${seq}-all-shot-*" -exec rm -rf {} + 2>/dev/null || true
}

for (( idx=0; idx<${#SEQUENCES[@]}; idx++ )); do
    SEQ_NAME="${SEQUENCES[$idx]}"
    echo ""
    echo "[$(date '+%H:%M:%S')] Step 3 [$((idx+1))/${#SEQUENCES[@]}]: Processing $SEQ_NAME"

    clear_caches "$SEQ_NAME"

    echo "  VIPE + HaMeR..."
    $PYTHON run_opt.py data=video_vipe "data.seq=$SEQ_NAME" run_opt=False run_vis=False 2>&1 | tail -3

    echo "  Optimizing (smooth_fit)..."
    $PYTHON run_opt.py data=video_vipe "data.seq=$SEQ_NAME" run_vis=False 2>&1 | tail -3

    # ── Per-chunk pair emission ──
    if [ -n "$EMIT_PAIRS_ROOT" ]; then
        TODAY="$(date +%Y-%m-%d)"
        CHUNK_LOG="$LOG_ROOT/video-custom/$TODAY/${SEQ_NAME}-all-shot-0-0--1"
        SMOOTH_NPZ="$CHUNK_LOG/smooth_fit/${SEQ_NAME}_000060_world_results.npz"
        HAMER_PKL="$TEST_DIR/dynhamr/hamer_out/${SEQ_NAME}/${SEQ_NAME}.pkl"
        IMG_DIR="$TEST_DIR/images/${SEQ_NAME}"

        if [ ! -f "$SMOOTH_NPZ" ]; then
            echo "  WARN: smooth_fit NPZ not found at $SMOOTH_NPZ, skipping pair emit"
        elif [ ! -f "$HAMER_PKL" ]; then
            echo "  WARN: HaMeR pkl not found at $HAMER_PKL, skipping pair emit"
        else
            # Per-chunk SLERP postprocess to get outlier mask + post NPZ
            CHUNK_RAW_NPZ="$CHUNK_LOG/smooth_fit/${SEQ_NAME}_raw.npz"
            CHUNK_POST_NPZ="$CHUNK_LOG/smooth_fit/${SEQ_NAME}_post.npz"
            cp "$SMOOTH_NPZ" "$CHUNK_RAW_NPZ"
            $PYTHON postprocess_npz.py --input "$SMOOTH_NPZ" --output "$CHUNK_POST_NPZ" 2>&1 | tail -3

            # Discover chunk's actual frame count (= smaller of MAX_FRAMES, image dir size)
            N_CHUNK=$(ls "$IMG_DIR" 2>/dev/null | wc -l)

            # Compute src_frame_start (chunks are 0-indexed in pair_writer)
            # SEQ_NAME = test60-pK ⇒ chunk_idx = K-1
            CHUNK_NUM=$(echo "$SEQ_NAME" | sed -E 's/.*-p([0-9]+)$/\1/')
            CHUNK_IDX=$((CHUNK_NUM - 1))
            FPS_F=$(echo "scale=8; $FPS_NUM / $FPS_DEN" | bc)
            SRC_FRAME_START=$(echo "scale=0; ($CHUNK_IDX * $MAX_FRAMES * $FPS_DEN + $FPS_NUM/2) / $FPS_NUM" | bc)
            # Actually: chunk_idx * (MAX_FRAMES seconds at 30fps assumed) → src frame at fps_real
            SRC_FRAME_START=$(echo "scale=0; $CHUNK_IDX * $MAX_FRAMES / 30.0 * $FPS_F / 1" | bc 2>/dev/null || echo $((CHUNK_IDX * MAX_FRAMES)))

            PAIR_GLOBAL_IDX=$((PAIR_GLOBAL_START + CHUNK_IDX))

            $PYTHON "$SCRIPT_DIR/scripts/pair_writer.py" \
                --source-video "$INPUT_VIDEO" \
                --chunk-idx "$CHUNK_IDX" \
                --src-frame-start "$SRC_FRAME_START" \
                --n-frames "$N_CHUNK" \
                --fps-src "$FPS_F" \
                --img-w "$WIDTH" \
                --img-h "$HEIGHT" \
                --hamer-pkl "$HAMER_PKL" \
                --image-dir "$IMG_DIR" \
                --dyn-post-npz "$CHUNK_POST_NPZ" \
                --dyn-raw-npz "$CHUNK_RAW_NPZ" \
                --out-root "$EMIT_PAIRS_ROOT" \
                --pair-global-idx "$PAIR_GLOBAL_IDX" 2>&1 | tail -5
        fi
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Merge chunks (or just copy if 1 chunk)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 4: Merge chunks"

LOG_DATE="$(date +%Y-%m-%d)"
LOG_DIR="$LOG_ROOT/video-custom/$LOG_DATE"

if (( ${#SEQUENCES[@]} > 1 )); then
    echo "  Merging ${#SEQUENCES[@]} chunks via merge_chunks.py..."
    $PYTHON "$SCRIPT_DIR/scripts/merge_chunks.py" \
        --seq "$SEQ" \
        --log-dir "$LOG_DIR" \
        --max-frames "$MAX_FRAMES" \
        --ffmpeg "$FFMPEG"
fi

# Find the merged result dir (single seq) or chunked-and-merged dir
RESULT_DIR=$(find "$LOG_DIR" -maxdepth 1 -type d -name "${SEQ}-all-shot-*" | sort | tail -n1)
if [ -z "$RESULT_DIR" ]; then
    echo "ERROR: Result directory not found for $SEQ"
    exit 1
fi
echo "  Result dir: $RESULT_DIR"

# Pick the smooth_fit final NPZ (after merge if chunked, original otherwise)
NPZ=$(find "$RESULT_DIR" -path "*smooth_fit*world_results.npz" | sort | tail -1)
echo "  Using NPZ: $NPZ"

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Postprocess (SLERP outlier fix)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 5: Postprocess NPZ (SLERP)"
$PYTHON postprocess_npz.py --input "$NPZ" --output "$OUTPUT_DIR/${SEQ}.npz" 2>&1 | tail -3
cp "$NPZ" "$OUTPUT_DIR/${SEQ}_raw.npz"

# Generate cameras.json from postprocessed NPZ
$PYTHON -c "
import numpy as np, json
d = np.load('$OUTPUT_DIR/${SEQ}.npz', allow_pickle=True)
cam_R = d['cam_R'][0]; cam_t = d['cam_t'][0]; intrins = d['intrins']
T = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
R_out = np.einsum('ij,bjk->bik', T, cam_R)
t_out = np.einsum('ij,bj->bi', T, cam_t)
N = len(R_out)
intrins_abs = np.abs(intrins).tolist()
cam_data = {
    'rotation': R_out.reshape(N, 9).tolist(),
    'translation': t_out.tolist(),
    'intrinsics': [intrins_abs] * N,
}
with open('$OUTPUT_DIR/cameras.json', 'w') as f:
    json.dump(cam_data, f, indent=1)
print(f'  cameras.json: {N} frames')
"

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Render (optional, full merged sequence)
# ──────────────────────────────────────────────────────────────────────────────
if [ "$DO_RENDER" = "true" ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Step 6: Render mesh video"

    # Render each chunk dir separately (full-merged dir is missing the unified
    # cameras/images/.hydra needed by run_vis), then ffmpeg-concat.
    CHUNK_DIRS=$(find "$LOG_DIR" -maxdepth 1 -type d -name "${SEQ}-p*-all-shot-*" | sort)
    if [ -z "$CHUNK_DIRS" ]; then
        # Single-chunk case: render the merged dir directly
        CHUNK_DIRS="$RESULT_DIR"
    fi

    RENDER_TMP="$OUTPUT_DIR/_chunk_renders"
    mkdir -p "$RENDER_TMP"
    CONCAT_LIST="$RENDER_TMP/concat.txt"
    : > "$CONCAT_LIST"

    for CD in $CHUNK_DIRS; do
        CN=$(basename "$CD" | sed 's/-all-shot.*//')
        OD="$RENDER_TMP/$CN"
        mkdir -p "$OD"
        $PYTHON run_vis.py --log_root "$CD" --save_root "$OD" \
            --phases smooth_fit --render_views src_cam --overwrite 2>&1 | tail -3
        MP=$(ls "$OD"/*src_cam.mp4 2>/dev/null | head -1)
        if [ -n "$MP" ]; then
            echo "file '$MP'" >> "$CONCAT_LIST"
        else
            echo "  WARN: no src_cam mp4 produced for $CN"
        fi
    done

    if [ -s "$CONCAT_LIST" ]; then
        $FFMPEG -y -f concat -safe 0 -i "$CONCAT_LIST" -c copy \
            "$OUTPUT_DIR/${SEQ}_mesh.mp4" 2>&1 | tail -2
    fi
else
    echo "[$(date '+%H:%M:%S')] Step 6: Render skipped (--no-render)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Cleanup
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Step 7: Cleanup intermediates"

# Clean per-chunk caches
for SEQ_NAME in "${SEQUENCES[@]}" "$SEQ"; do
    clear_caches "$SEQ_NAME"
    rm -f "$VIDEO_DIR/${SEQ_NAME}.mp4"
done

# Clean intermediate optimization output (keep merged result npz)
find "$LOG_DIR" -maxdepth 1 -type d -name "${SEQ}-p*-all-shot-*" -exec rm -rf {} + 2>/dev/null || true
# Optional: clean root_fit checkpoints to save space
find "$RESULT_DIR" -name "*.pth" -delete 2>/dev/null || true
find "$RESULT_DIR/root_fit" -type f -delete 2>/dev/null || true
rmdir "$RESULT_DIR/root_fit" 2>/dev/null || true

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "Output: $OUTPUT_DIR/"
echo "============================================"
ls -lh "$OUTPUT_DIR/"
echo ""
du -sh "$OUTPUT_DIR/"
