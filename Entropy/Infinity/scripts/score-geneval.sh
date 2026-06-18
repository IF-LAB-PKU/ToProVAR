#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFINITY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_REPO_ROOT="$(cd "${INFINITY_DIR}/../.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-toprovar_work}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace}"
CONTAINER_INFINITY_DIR="${CONTAINER_REPO_ROOT}/Entropy/Infinity"
HOST_GENEVAL_DIR="${INFINITY_DIR}/evaluation/gen_eval"
HOST_VENDOR_DIR_DEFAULT="${HOST_GENEVAL_DIR}/vendor_mmdet2"
CONTAINER_VENDOR_DIR_DEFAULT="${CONTAINER_INFINITY_DIR}/evaluation/gen_eval/vendor_mmdet2"
HOST_LEGACY_CONFIG_DEFAULT="${HOST_GENEVAL_DIR}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
HOST_GENEVAL_OPENCLIP_DEFAULT="${HOST_GENEVAL_DIR}/models/open_clip_pytorch_model.bin"

resolve_repo_path() {
    local raw_path="$1"
    if [[ "${raw_path}" = /* ]]; then
        readlink -m "${raw_path}"
    else
        readlink -m "${HOST_REPO_ROOT}/${raw_path}"
    fi
}

map_host_path_to_container() {
    local host_path="$1"
    case "${host_path}" in
        "${HOST_REPO_ROOT}")
            printf '%s\n' "${CONTAINER_REPO_ROOT}"
            ;;
        "${HOST_REPO_ROOT}"/*)
            printf '%s%s\n' "${CONTAINER_REPO_ROOT}" "${host_path#${HOST_REPO_ROOT}}"
            ;;
        *)
            echo "Path must stay inside ${HOST_REPO_ROOT}: ${host_path}" >&2
            echo "This wrapper only supports repo-local paths because the container mounts the repo at ${CONTAINER_REPO_ROOT}." >&2
            exit 1
            ;;
    esac
}

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
    echo "Container ${CONTAINER_NAME} is not running." >&2
    exit 1
fi

image_root_host="$(resolve_repo_path "${1:-${IMAGE_ROOT_PATH:-${INFINITY_DIR}/outputs/gen_eval_official/images}}")"
output_root_host="$(dirname "${image_root_host}")"
results_path_host="$(resolve_repo_path "${RESULTS_PATH:-${output_root_host}/results.jsonl}")"
summary_path_host="$(resolve_repo_path "${SUMMARY_PATH:-${output_root_host}/summary.txt}")"
model_path="${MODEL_PATH:-/data/models/mask2former}"
model_config_in_container="${MODEL_CONFIG_IN_CONTAINER:-/opt/conda/envs/dpg/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py}"
vendor_dir_host="${GENEVAL_VENDOR_DIR:-${HOST_VENDOR_DIR_DEFAULT}}"
vendor_dir_container=""
hf_endpoint="${HF_ENDPOINT:-https://huggingface.co}"
hf_hub_disable_xet="${HF_HUB_DISABLE_XET:-1}"
geneval_openclip_pretrained_raw="${GENEVAL_OPENCLIP_PRETRAINED:-}"
if [[ -z "${geneval_openclip_pretrained_raw}" && -f "${HOST_GENEVAL_OPENCLIP_DEFAULT}" ]]; then
    geneval_openclip_pretrained_raw="${HOST_GENEVAL_OPENCLIP_DEFAULT}"
fi
geneval_openclip_pretrained_in_container="${geneval_openclip_pretrained_raw}"

if [[ ! -d "${image_root_host}" ]]; then
    echo "Image root path does not exist: ${image_root_host}" >&2
    exit 1
fi

if [[ -d "${vendor_dir_host}" && -d "${vendor_dir_host}/mmdet" && -d "${vendor_dir_host}/mmcv" ]]; then
    vendor_dir_host="$(resolve_repo_path "${vendor_dir_host}")"
    vendor_dir_container="$(map_host_path_to_container "${vendor_dir_host}")"
    if [[ -z "${MODEL_CONFIG:-}" ]]; then
        legacy_config_host="$(resolve_repo_path "${HOST_LEGACY_CONFIG_DEFAULT}")"
        model_config_in_container="$(map_host_path_to_container "${legacy_config_host}")"
    fi
fi

if [[ -n "${MODEL_CONFIG:-}" ]]; then
    model_config_host="$(resolve_repo_path "${MODEL_CONFIG}")"
    if [[ ! -f "${model_config_host}" ]]; then
        echo "Model config file does not exist: ${model_config_host}" >&2
        exit 1
    fi
    model_config_in_container="$(map_host_path_to_container "${model_config_host}")"
fi

if [[ -n "${geneval_openclip_pretrained_raw}" ]]; then
    if [[ "${geneval_openclip_pretrained_raw}" = /* && -f "${geneval_openclip_pretrained_raw}" ]]; then
        geneval_openclip_pretrained_raw="$(resolve_repo_path "${geneval_openclip_pretrained_raw}")"
        geneval_openclip_pretrained_in_container="$(map_host_path_to_container "${geneval_openclip_pretrained_raw}")"
    elif [[ -f "${HOST_REPO_ROOT}/${geneval_openclip_pretrained_raw}" ]]; then
        geneval_openclip_pretrained_raw="$(resolve_repo_path "${geneval_openclip_pretrained_raw}")"
        geneval_openclip_pretrained_in_container="$(map_host_path_to_container "${geneval_openclip_pretrained_raw}")"
    fi
fi

image_root_container="$(map_host_path_to_container "${image_root_host}")"
results_path_container="$(map_host_path_to_container "${results_path_host}")"
summary_path_container="$(map_host_path_to_container "${summary_path_host}")"

echo "Running GenEval scoring in container ${CONTAINER_NAME}"
echo "Host image dir: ${image_root_host}"
echo "Host results:   ${results_path_host}"
echo "Host summary:   ${summary_path_host}"
echo "OpenCLIP pretrained: ${geneval_openclip_pretrained_in_container}"

docker exec \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e HF_ENDPOINT="${hf_endpoint}" \
    -e HF_HUB_DISABLE_XET="${hf_hub_disable_xet}" \
    -e INFINITY_DIR_IN_CONTAINER="${CONTAINER_INFINITY_DIR}" \
    -e IMAGE_ROOT_PATH_IN_CONTAINER="${image_root_container}" \
    -e RESULTS_PATH_IN_CONTAINER="${results_path_container}" \
    -e SUMMARY_PATH_IN_CONTAINER="${summary_path_container}" \
    -e MODEL_CONFIG_IN_CONTAINER="${model_config_in_container}" \
    -e MODEL_PATH="${model_path}" \
    -e GENEVAL_VENDOR_DIR_IN_CONTAINER="${vendor_dir_container}" \
    -e GENEVAL_OPENCLIP_PRETRAINED="${geneval_openclip_pretrained_in_container}" \
    "${CONTAINER_NAME}" \
    bash -lc '
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate dpg

mkdir -p "$(dirname "${RESULTS_PATH_IN_CONTAINER}")"
cd "${INFINITY_DIR_IN_CONTAINER}"
if [[ -n "${GENEVAL_VENDOR_DIR_IN_CONTAINER}" ]]; then
    export PYTHONPATH="${GENEVAL_VENDOR_DIR_IN_CONTAINER}:${INFINITY_DIR_IN_CONTAINER}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${INFINITY_DIR_IN_CONTAINER}:${PYTHONPATH:-}"
fi

python evaluation/gen_eval/evaluate_images.py \
    --imagedir "${IMAGE_ROOT_PATH_IN_CONTAINER}" \
    --outfile "${RESULTS_PATH_IN_CONTAINER}" \
    --model-config "${MODEL_CONFIG_IN_CONTAINER}" \
    --model-path "${MODEL_PATH}" \
    --options model=mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco \
    | tee "${SUMMARY_PATH_IN_CONTAINER%.txt}_eval.log"

python evaluation/gen_eval/summary_scores.py "${RESULTS_PATH_IN_CONTAINER}" | tee "${SUMMARY_PATH_IN_CONTAINER}"
'

echo "GenEval results written to: ${results_path_host}"
echo "GenEval summary written to: ${summary_path_host}"
