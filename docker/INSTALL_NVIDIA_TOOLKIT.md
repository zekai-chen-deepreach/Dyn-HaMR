# Installing NVIDIA Container Toolkit (host prerequisite)

`docker run --gpus all` won't work without NVIDIA Container Toolkit. The local
smoke test needs this on the host (Ubuntu/Debian).

Run these once on the host:

```bash
# Add NVIDIA repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Wire it into Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see `nvidia-smi` output showing your GPU. Then the smoke test
container can use `--gpus all`.

(AWS Batch GPU AMIs already have this configured; this is only for local dev.)
