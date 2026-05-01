# Dyn-HaMR pair-data Docker image

Self-contained image that runs `run_pipeline_chunked.sh` on one input video and
emits HaMeR↔DynHaMR paired NPZ files (the dataset for downstream feed-forward
training).

## What goes in the image

- Ubuntu 22.04 + CUDA 12.1
- Miniconda + the `dynhamr` conda env (PyTorch 1.13 cu117, dyn-hamr, hamer, vipe)
- Repo: `zekai-chen-deepreach/Dyn-HaMR` @ `main` with submodules pinned to `dynhamr-compat`
- All checkpoints baked in:
  - `_DATA/` (~7.6 GB): MANO, HaMeR, ViTPose, HMP priors
  - `pretrained_models/detector.pt` (~52 MB): YOLO hand detector (WiLoR)
  - Torch hub cache (~6 GB): DROID-SLAM, GroundingDINO, depth-anything, etc.
- Total image size: ~14 GB

## Build

```bash
cd /path/to/Dyn-HaMR
bash docker/build.sh                    # local build, tag dynhamr-pair:latest
bash docker/build.sh --push <ECR_URI>   # build + push to ECR
```

`build.sh` first stages `_DATA/` and the torch hub cache as zstd tarballs in the
build context (so the COPY layer is one layer, easy to invalidate).

## Smoke test locally

```bash
docker run --rm --gpus all \
    -e VIDEO_S3=s3://your-bucket/raw/test.mp4 \
    -e PAIR_OUT_S3=s3://your-bucket/pair_dataset/ \
    -e PAIR_GLOBAL_START=0 \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=us-east-1 \
    dynhamr-pair:latest
```

Output S3 layout:

```
s3://your-bucket/pair_dataset/
├── shards/
│   └── 00000-00999/
│       ├── 00000_<video_stem>__chunk_0000.npz
│       ├── 00001_<video_stem>__chunk_0001.npz
│       └── …
└── manifests/
    ├── manifests_<job_id_1>.jsonl   # one row per emitted pair
    ├── manifests_<job_id_2>.jsonl
    └── …
```

After all jobs complete, run a one-shot aggregator (not in this image) to merge
the per-job `manifests/*.jsonl` into a single `manifest.parquet`.

## AWS Batch wiring

- **Job definition**: container image = ECR URI; vCPUs = 4; memory = 24 GB; GPU = 1.
- **Compute environment**: GPU instance types (`g4dn.2xlarge` is plenty;
  `g5.xlarge` is a bit faster). Spot is fine — the pipeline is idempotent at the
  per-video grain.
- **Job parameters** (`Ref::VIDEO_S3` etc. forwarded to env):
  - `VIDEO_S3` — s3 URI of one source video
  - `PAIR_OUT_S3` — s3 prefix for outputs
  - `PAIR_GLOBAL_START` — global pair index for this video's first chunk
- A separate `submit_jobs.py` (not in this image) lists S3 source videos, assigns
  `PAIR_GLOBAL_START` slots (e.g. `chunk_count_estimate × video_idx`), and calls
  `batch.submit_job` once per video.

## Iteration tips

- If you change only `run_pipeline_chunked.sh`, `pair_writer.py`, or other repo
  code, rebuild is fast: the heavy `_DATA` and `torch_hub` COPY layers are
  cached, only the final `git clone` + pip layer (without checkpoints) is reinvalidated.
- If you change checkpoints, delete `docker/_DATA.tar.zst` to force a re-stage.
- If the submodule patches (hamer/vipe `dynhamr-compat`) move forward, push them
  to GitHub and rebuild — the Dockerfile pins by branch name.
