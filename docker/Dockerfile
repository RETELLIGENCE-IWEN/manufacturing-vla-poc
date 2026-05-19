FROM maniskill/base:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ManiSkill asset/data location inside the mounted repo
ENV MS_ASSET_DIR=/workspace/manufacturing-vla-poc/.maniskill/data
ENV MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# Useful for NVIDIA container runtime + rendering/Vulkan paths
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/manufacturing-vla-poc

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      pyyaml \
      rich \
      imageio \
      imageio-ffmpeg

CMD ["/bin/bash"]