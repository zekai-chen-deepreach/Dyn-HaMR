# Dyn-HaMR Pipeline

4D hand motion reconstruction from monocular video. Outputs MANO hand pose parameters (NPZ) + mesh rendering.

## Quick Start

```bash
# Put your video in test/videos/ (H.264 codec, 1080p)
cp my_video.mp4 test/videos/my-video.mp4

# Run full pipeline (VIPE → HaMeR → optimize → postprocess → render)
bash run_pipeline.sh my-video

# Output: test/output/my-video/
#   ├── my-video_postprocessed.npz   # Final hand pose
#   ├── my-video_raw.npz             # Raw optimization result
#   ├── my-video_mesh.mp4            # Mesh overlay video
#   ├── images/                      # Extracted frames (for re-rendering)
#   ├── cameras/                     # Camera params (for re-rendering)
#   ├── track_preds/                 # Hand detection (for re-rendering)
#   ├── shot_idcs.json               # Shot segmentation
#   └── hydra_config/                # Config snapshot
```

## Environment

Two conda environments:

| Environment | Purpose |
|------------|---------|
| `dynhamr` | HaMeR hand detection + optimization + rendering |
| `vipe` | VIPE camera estimation (DROID-SLAM variant) |

## Input Requirements

- **Format**: MP4, H.264 codec (HEVC will be auto-converted)
- **Resolution**: 1080p (4K will need manual downscale: `ffmpeg -i in.mp4 -vf scale=1920:1080 -c:v libx264 -an out.mp4`)
- **Length**: <700 frames recommended. VIPE may OOM on longer videos.
- **Placement**: `test/videos/<seq>.mp4`

## Output Structure

```
test/output/<seq>/
├── <seq>_postprocessed.npz    # Final NPZ (vertex-space outliers fixed)
├── <seq>_raw.npz              # Raw optimization NPZ
├── <seq>_mesh.mp4             # Mesh overlay rendering (src_cam view)
├── images/                    # Extracted video frames
├── cameras/                   # VIPE camera parameters
├── track_preds/               # HaMeR hand detection results
├── shot_idcs.json             # Shot segmentation index
└── hydra_config/              # Hydra config for reproducibility
```

All files needed for re-rendering are included in the output directory.

## NPZ Format

| Key | Shape | Description |
|-----|-------|-------------|
| `trans` | (2, T, 3) | Wrist translation per hand |
| `root_orient` | (2, T, 3) | Global hand orientation (axis-angle) |
| `pose_body` | (2, T, 15, 3) | Finger joint poses (axis-angle) |
| `betas` | (2, 10) | Hand shape parameters |
| `is_right` | (2, T) | 0=left hand, 1=right hand |
| `cam_R` | (2, T, 3, 3) | Camera rotation (world-to-camera) |
| `cam_t` | (2, T, 3) | Camera translation |
| `intrins` | (4,) | Camera intrinsics [fx, fy, cx, cy] |

## Pipeline Steps (run_pipeline.sh)

1. **VIPE** — Camera estimation via DROID-SLAM
2. **HaMeR** — YOLO hand detection + HaMeR mesh estimation
3. **Optimization** — root_fit (50 iter) + smooth_fit (60 iter) with axis-angle [-π,π] normalization
4. **Post-processing** — Vertex-space outlier detection + parameter interpolation (skips first/last 1s)
5. **Rendering** — Mesh overlay on source video frames

## Re-rendering from Output

If you have an output directory, you can re-render without re-running the pipeline:

```bash
conda activate dynhamr
cd dyn-hamr

# The output directory contains everything needed
python run_vis.py \
    --log_root <output_dir>/hydra_config/.. \
    --save_root <render_output_dir> \
    --phases smooth_fit \
    --render_views src_cam \
    --overwrite
```

## Key Optimizations

- **Axis-angle normalization**: Wraps root_orient and finger joints to [-π,π] after each LBFGS step, preventing rotation flips
- **Vertex-space post-processing**: Detects abnormal mesh jumps via MANO forward pass, fixes parameters by interpolation
- **Auto H.264 conversion**: HEVC videos are auto-converted (VIPE OOMs on HEVC due to higher memory usage)
- **60-iteration smooth_fit**: Reduced from 300 (98% loss convergence, 5x faster)

## Hardware Requirements

| Component | Pipeline (full) | Render only |
|-----------|----------------|-------------|
| GPU VRAM | 15 GB (T4) | 2 GB |
| RAM | 64 GB | 16 GB |
| Instance | g4dn.4xlarge | g4dn.xlarge |

> **RAM note**: VIPE SLAM requires ~50GB RAM for 400-frame videos. 32GB is insufficient for >300 frames.

## Directory Structure

```
Dyn-HaMR/
├── run_pipeline.sh              # One-command full pipeline
├── dyn-hamr/
│   ├── run_opt.py               # Main optimization entry
│   ├── run_vis.py               # Mesh rendering
│   ├── render_skeleton.py       # Skeleton overlay rendering
│   ├── postprocess_npz.py       # NPZ vertex-space post-processing
│   ├── confs/
│   │   ├── config.yaml
│   │   ├── optim.yaml           # Optimization parameters
│   │   └── data/video_vipe.yaml # Data paths config
│   └── optim/
│       └── optimizers.py        # LBFGS optimizer + axis-angle normalization
├── third-party/
│   ├── hamer/                   # HaMeR hand estimation
│   └── vipe/                    # VIPE camera estimation
├── test/
│   ├── videos/                  # INPUT: source videos
│   └── output/                  # OUTPUT: results per video
├── outputs/logs/                # Intermediate optimization logs
└── _DATA/                       # Model weights (MANO, HaMeR)
```
