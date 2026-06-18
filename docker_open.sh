#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONTAINER_NAME="${CONTAINER_NAME:-toprovar_dev}"
DOCKER_IMAGE="${DOCKER_IMAGE:-toprovar:latest}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/data}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/data}"

if [[ ! -d "${HOST_DATA_DIR}" ]]; then
  echo "HOST_DATA_DIR does not exist: ${HOST_DATA_DIR}" >&2
  echo "Set HOST_DATA_DIR=/path/to/models-root or create the directory before starting the container." >&2
  exit 1
fi

docker run -it --gpus all --network host \
  -v "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
  -v "${REPO_ROOT}:/workspace" \
  --name "${CONTAINER_NAME}" "${DOCKER_IMAGE}" bash
