# Dyn-HaMR pair-data generator image for AWS Batch
#
# Build:
#   docker build -t dynhamr-pair:latest -f docker/Dockerfile .
#
# Run locally (smoke test):
#   docker run --rm --gpus all \
#       -e VIDEO_S3=s3://your-bucket/raw/test.mp4 \
#       -e PAIR_OUT_S3=s3://your-bucket/pair_dataset/ \
#       -e PAIR_GLOBAL_START=0 \
#       -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=us-east-1 \
#       dynhamr-pair:latest
#
# Notes:
# - We bake _DATA/ + torch hub caches into the image so containers start fast.
#   That makes the image ~14 GB. If you want a slimmer image, mount these from
#   a persistent EBS volume or pull from S3 at startup instead.
# - The runtime expects: NVIDIA GPU, CUDA 12.x driver on host (Batch GPU AMI handles this).

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc ca-certificates curl ffmpeg git libegl1 libgl1 libglib2.0-0 \
        libosmesa6-dev libsm6 libxext6 libxrender1 unzip wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 for S3 IO
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip

# ── Miniconda + dynhamr conda env ───────────────────────────────────────────
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh \
    && /opt/conda/bin/conda init bash

# Repo + submodules. Pinning by SHA in production; using main here.
ARG DYN_HAMR_REF=main
RUN git clone --recursive --depth 1 -b ${DYN_HAMR_REF} \
        https://github.com/zekai-chen-deepreach/Dyn-HaMR.git /workspace
WORKDIR /workspace

# Pin submodules to the dynhamr-compat branches on the user's forks (the
# patches that add --no_viz, fast-pose-only path, bbox-in-pkl, etc. live there).
# .gitmodules points at upstream (ZhengdiYu/hamer, nv-tlabs/vipe), so add the
# fork remotes here.
RUN cd third-party/hamer \
    && git remote add fork https://github.com/zekai-chen-deepreach/hamer.git \
    && git fetch fork dynhamr-compat \
    && git checkout -B dynhamr-compat FETCH_HEAD \
    && cd ../vipe \
    && git remote add fork https://github.com/zekai-chen-deepreach/vipe.git \
    && git fetch fork dynhamr-compat \
    && git checkout -B dynhamr-compat FETCH_HEAD

# ── Python env ──────────────────────────────────────────────────────────────
SHELL ["/bin/bash", "-lc"]

# Recent miniconda requires explicit ToS acceptance for the default channels.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n dynhamr python=3.10 -y \
    && conda run -n dynhamr pip install --no-cache-dir \
        torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117 \
    # torch 1.13's cpp_extension imports pkg_resources, which setuptools >=70
    # removed. Pin to 59.5.0 (matches requirements.txt) before building extensions.
    && conda run -n dynhamr pip install --no-cache-dir 'setuptools==59.5.0' wheel \
    # Pre-install build/runtime deps that chumpy's setup.py imports at build time
    # (it does `import pip`, `import numpy`). Otherwise PEP-517 isolated builds fail.
    && conda run -n dynhamr pip install --no-cache-dir --no-build-isolation \
        'numpy<1.24' pip \
    && conda run -n dynhamr pip install --no-cache-dir --no-build-isolation \
        torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html \
    && conda run -n dynhamr pip install --no-cache-dir --no-build-isolation \
        -r requirements.txt \
    && conda run -n dynhamr pip install --no-cache-dir -e .

# HaMeR (with ViTPose) + DROID-SLAM (only DROID-SLAM optional path; VIPE replaces it)
RUN cd third-party/hamer \
    && conda run -n dynhamr pip install --no-cache-dir -e ".[all]" || true \
    && cd third-party/ViTPose && conda run -n dynhamr pip install --no-cache-dir -v -e . || true

# VIPE (separate conda env per upstream README, but shared with dynhamr works in our pipeline)
RUN cd /workspace/third-party/vipe \
    && conda run -n dynhamr pip install --no-cache-dir -e . || true

# ── Model checkpoints ───────────────────────────────────────────────────────
# Strategy: copy from build context (must be staged by build helper) into the image.
# Build helper script: docker/build.sh tars _DATA/ and the torch hub cache, then COPY here.
COPY docker/_DATA.tar.zst /tmp/_DATA.tar.zst
RUN mkdir -p /workspace/_DATA \
    && zstd -d /tmp/_DATA.tar.zst -o /tmp/_DATA.tar \
    && tar -xf /tmp/_DATA.tar -C /workspace \
    && rm /tmp/_DATA.tar /tmp/_DATA.tar.zst

# HaMeR YOLO detector
COPY docker/detector.pt /workspace/third-party/hamer/pretrained_models/detector.pt

# Torch hub (DROID-SLAM weights, GroundingDINO, depth, etc., used by VIPE)
COPY docker/torch_hub.tar.zst /tmp/torch_hub.tar.zst
RUN mkdir -p /root/.cache/torch/hub \
    && zstd -d /tmp/torch_hub.tar.zst -o /tmp/torch_hub.tar \
    && tar -xf /tmp/torch_hub.tar -C /root/.cache/torch/ \
    && rm /tmp/torch_hub.tar /tmp/torch_hub.tar.zst

# ── Runtime config ──────────────────────────────────────────────────────────
ENV PYOPENGL_PLATFORM=egl \
    EGL_DEVICE_ID=0 \
    DISPLAY=

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
