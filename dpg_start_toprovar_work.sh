#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-toprovar_work}"
CONDA_ENV="${CONDA_ENV:-dpg}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace}"

echo ">>> Starting container: ${CONTAINER_NAME}"
docker start "${CONTAINER_NAME}"

echo ">>> Opening shell in conda env: ${CONDA_ENV}"
docker exec -it "${CONTAINER_NAME}" /bin/bash -c "
    echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${CONTAINER_REPO_ROOT}' > /tmp/env_init.sh && \
    /bin/bash --rcfile /tmp/env_init.sh"
