#!/usr/bin/env bash
# Stage heavy assets, then build the docker image.
#
# We copy the local _DATA tree, the YOLO detector.pt, and the torch hub cache
# (DROID-SLAM weights, GroundingDINO, depth checkpoints used by VIPE) into
# the build context. These can't be fetched at build time:
#   - MANO_RIGHT.pkl    requires manual signup
#   - droid.pth         lives on Google Drive (gdown), unreliable in CI
#   - VIPE auxiliary    weights are downloaded on first run; baking them in
#                       removes a 5-7 minute cold-start hit per job
#
# Usage:
#   bash docker/build.sh [--push <ECR_URI>]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"
TAG="dynhamr-pair:latest"
PUSH_TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH_TARGET="$2"; shift 2 ;;
        --tag)  TAG="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "[build] staging _DATA → $DOCKER_DIR/_DATA.tar.zst"
if [ ! -f "$DOCKER_DIR/_DATA.tar.zst" ] || [ "$REPO_ROOT/_DATA" -nt "$DOCKER_DIR/_DATA.tar.zst" ]; then
    tar -C "$REPO_ROOT" -cf - _DATA \
        | zstd -T0 -10 -o "$DOCKER_DIR/_DATA.tar.zst"
fi

echo "[build] staging detector.pt"
cp -u "$REPO_ROOT/third-party/hamer/pretrained_models/detector.pt" "$DOCKER_DIR/detector.pt"

echo "[build] staging torch hub cache → $DOCKER_DIR/torch_hub.tar.zst"
HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
if [ ! -d "$HUB_DIR" ]; then
    echo "[build] WARNING: $HUB_DIR not found. VIPE will download weights at first run."
    # create empty tarball so COPY doesn't fail
    tar -cf - --files-from /dev/null | zstd -o "$DOCKER_DIR/torch_hub.tar.zst"
elif [ ! -f "$DOCKER_DIR/torch_hub.tar.zst" ] || [ "$HUB_DIR" -nt "$DOCKER_DIR/torch_hub.tar.zst" ]; then
    tar -C "$(dirname "$HUB_DIR")" -cf - "$(basename "$HUB_DIR")" \
        | zstd -T0 -10 -o "$DOCKER_DIR/torch_hub.tar.zst"
fi

echo "[build] sizes:"
ls -lh "$DOCKER_DIR/_DATA.tar.zst" "$DOCKER_DIR/torch_hub.tar.zst" "$DOCKER_DIR/detector.pt"

echo "[build] docker build -t $TAG"
docker build -t "$TAG" -f "$DOCKER_DIR/Dockerfile" "$REPO_ROOT"

if [ -n "$PUSH_TARGET" ]; then
    echo "[build] tagging and pushing to $PUSH_TARGET"
    docker tag "$TAG" "$PUSH_TARGET"
    aws ecr get-login-password | docker login --username AWS --password-stdin "${PUSH_TARGET%%/*}"
    docker push "$PUSH_TARGET"
fi

echo "[build] done. Image: $TAG"
