#!/bin/bash
set -euo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-toprovar:latest}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

docker build -t "${DOCKER_IMAGE}" "${REPO_ROOT}"
