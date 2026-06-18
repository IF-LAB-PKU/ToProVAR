FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]
