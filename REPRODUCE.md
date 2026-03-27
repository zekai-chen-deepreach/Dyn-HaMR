# Dyn-HaMR Pipeline

4D hand motion reconstruction from monocular video. Outputs MANO hand pose parameters (NPZ).

## Quick Start

```bash
# Put your video in test/videos/
cp my_video.mp4 test/videos/my-video.mp4

# Run full pipeline
bash run_pipeline.sh my-video

# Output: test/my-video_postprocessed.npz
```

## Environment

Two conda environments required:

| Environment | Purpose |
|------------|---------|
| `dynhamr` | HaMeR hand detection + optimization + visualization |
| `vipe` | VIPE camera estimation (DROID-SLAM variant) |

## Pipeline Steps

### Input
- `test/videos/<seq>.mp4` — 1080p video, H.264 codec, <700 frames recommended
- 4K videos: downscale to 1080p first (`ffmpeg -i input.mp4 -vf scale=1920:1080 -c:v libx264 -an output.mp4`)
- Long videos (>700 frames): may OOM during VIPE. Trim or split first.

### Step 1: VIPE Camera Estimation
```bash
conda activate vipe
cd third-party/vipe
vipe infer test/videos/<seq>.mp4 -p dynhamr
```
Outputs camera pose + intrinsics to `third-party/vipe/vipe_results/`.

### Step 2: HaMeR Hand Detection
```bash
conda activate dynhamr
cd dyn-hamr
python run_opt.py data=video_vipe data.seq=<seq> run_opt=False run_vis=False
```
Extracts frames, runs YOLO hand detector + HaMeR mesh estimation, loads VIPE cameras.

### Step 3: Optimization
```bash
python run_opt.py data=video_vipe data.seq=<seq> run_vis=False
```
- `root_fit` (50 iterations): global translation + orientation
- `smooth_fit` (60 iterations): full pose + shape + temporal smoothing
- Axis-angle normalization: wraps root_orient and finger joints to [-π, π] after each LBFGS step to prevent rotation flips

### Step 4: Post-processing
```bash
python postprocess_npz.py --input <npz_path> --output test/<seq>_postprocessed.npz
```
- Runs MANO forward pass to get vertices
- Detects frames with abnormal vertex displacement (MAD outlier detection)
- Interpolates outlier frames' parameters (trans, root_orient, pose_body) from neighbors
- Skips first/last 1 second (30 frames) to avoid boundary artifacts

### Output

**NPZ file** containing:
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

### Optional: Visualization
```bash
# Mesh overlay on source video
python run_vis.py --log_root <log_dir> --save_root <output_dir> --phases smooth_fit --render_views src_cam --overwrite

# Skeleton overlay (from postprocessed NPZ)
python render_skeleton.py --npz_path <npz> --video_path <video> --output_path <output.mp4>
```

## Directory Structure

```
Dyn-HaMR/
├── run_pipeline.sh              # One-command full pipeline
├── dyn-hamr/
│   ├── run_opt.py               # Main optimization entry
│   ├── run_vis.py               # Visualization (mesh rendering)
│   ├── render_skeleton.py       # Skeleton overlay rendering
│   ├── postprocess_npz.py       # NPZ post-processing
│   ├── confs/
│   │   ├── config.yaml
│   │   ├── optim.yaml           # Optimization parameters
│   │   └── data/video_vipe.yaml # Data paths config
│   ├── optim/
│   │   ├── optimizers.py        # LBFGS optimizer + axis-angle normalization
│   │   └── losses.py            # Loss functions
│   └── vis/
│       ├── output.py            # Rendering pipeline
│       └── tools.py             # OneEuroFilter, smoothing utilities
├── third-party/
│   ├── hamer/                   # HaMeR hand estimation
│   └── vipe/                    # VIPE camera estimation
│       └── vipe_results/        # VIPE output (auto-generated)
├── test/
│   ├── videos/<seq>.mp4         # INPUT: source videos
│   ├── <seq>_postprocessed.npz  # OUTPUT: final hand pose
│   ├── images/<seq>/            # Extracted frames (auto)
│   └── dynhamr/                 # Intermediate data (auto)
├── outputs/logs/                # Optimization logs + raw NPZ
└── _DATA/                       # Model weights (MANO, HaMeR)
```

## Key Parameters (optim.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `smooth.num_iters` | 60 | Smooth optimization iterations |
| `root.num_iters` | 50 | Root optimization iterations |
| `joints2d_sigma` | 100 | GMOF robust loss sigma |
| `joints3d_smooth` | [1000, 10000, 0] | Temporal smoothness weight per stage |
| `pose_prior` | [1, 1, 1] | Pose regularization weight |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 15 GB (T4) | 16+ GB |
| RAM | 32 GB | 32+ GB |
| VIPE needs RAM for SLAM — 16GB is not enough for >300 frame videos. |
