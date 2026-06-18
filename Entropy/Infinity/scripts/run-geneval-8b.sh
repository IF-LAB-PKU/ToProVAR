#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFINITY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_REPO_ROOT="$(cd "${INFINITY_DIR}/../.." && pwd)"
HOST_GENEVAL_DIR="${INFINITY_DIR}/evaluation/gen_eval"

CONTAINER_NAME="${CONTAINER_NAME:-toprovar_work}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace}"
CONTAINER_INFINITY_DIR="${CONTAINER_REPO_ROOT}/Entropy/Infinity"

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

metadata_file_host="$(resolve_repo_path "${METADATA_FILE:-${HOST_GENEVAL_DIR}/prompts/evaluation_metadata.jsonl}")"
run_root_host="$(resolve_repo_path "${OUT_ROOT:-${INFINITY_DIR}/outputs/gen_eval_official}")"
out_dir_host="$(resolve_repo_path "${OUT_DIR:-${run_root_host}/images}")"
n_samples="${N_SAMPLES:-4}"
seed="${SEED:-42}"
use_adaptive_scale="${USE_ADAPTIVE_SCALE:-0}"
adaptive_entropy_collect_start_scale_index="${ADAPTIVE_ENTROPY_COLLECT_START_SCALE_INDEX:-7}"
adaptive_entropy_trigger_start_scale_index="${ADAPTIVE_ENTROPY_TRIGGER_START_SCALE_INDEX:-8}"
adaptive_entropy_gamma="${ADAPTIVE_ENTROPY_GAMMA:-1.4e-6}"
adaptive_entropy_lambda="${ADAPTIVE_ENTROPY_LAMBDA:-1.0}"
adaptive_entropy_score_mode="${ADAPTIVE_ENTROPY_SCORE_MODE:-delta_h_width_power}"
adaptive_entropy_signal_mode="${ADAPTIVE_ENTROPY_SIGNAL_MODE:-layer_mean_delta}"
adaptive_entropy_signal_layer_idx="${ADAPTIVE_ENTROPY_SIGNAL_LAYER_IDX:-5}"
adaptive_max_raw_start_scale_index="${ADAPTIVE_MAX_RAW_START_SCALE_INDEX:-10}"
structure_layer_ratio="${STRUCTURE_LAYER_RATIO:-0.6}"
svd_source_scale_index="${SVD_SOURCE_SCALE_INDEX:-8}"
toprovar_entropy_side_channel_all="${TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL:-1}"
toprovar_entropy_side_channel_debug="${TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG:-0}"
toprovar_low_entropy_prune_ratio="${TOPROVAR_LOW_ENTROPY_PRUNE_RATIO:-}"
toprovar_high_entropy_prune_ratio="${TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO:-}"
toprovar_token_high_entropy_fraction_schedule="${TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE:-}"

if [[ ! -f "${metadata_file_host}" ]]; then
    echo "Metadata file does not exist: ${metadata_file_host}" >&2
    exit 1
fi

metadata_file_container="$(map_host_path_to_container "${metadata_file_host}")"
out_dir_container="$(map_host_path_to_container "${out_dir_host}")"

echo "Running GenEval generation in container ${CONTAINER_NAME}"
echo "Host metadata file: ${metadata_file_host}"
echo "Host output dir:    ${out_dir_host}"
echo "Entropy prune ratios: low=${toprovar_low_entropy_prune_ratio:-default}, high=${toprovar_high_entropy_prune_ratio:-default}"
echo "Token high-entropy fraction schedule: ${toprovar_token_high_entropy_fraction_schedule:-default}"

docker exec \
    -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e INFINITY_DIR_IN_CONTAINER="${CONTAINER_INFINITY_DIR}" \
    -e METADATA_FILE_IN_CONTAINER="${metadata_file_container}" \
    -e OUT_DIR_IN_CONTAINER="${out_dir_container}" \
    -e SEED="${seed}" \
    -e N_SAMPLES="${n_samples}" \
    -e USE_ADAPTIVE_SCALE="${use_adaptive_scale}" \
    -e ADAPTIVE_ENTROPY_COLLECT_START_SCALE_INDEX="${adaptive_entropy_collect_start_scale_index}" \
    -e ADAPTIVE_ENTROPY_TRIGGER_START_SCALE_INDEX="${adaptive_entropy_trigger_start_scale_index}" \
    -e ADAPTIVE_ENTROPY_GAMMA="${adaptive_entropy_gamma}" \
    -e ADAPTIVE_ENTROPY_LAMBDA="${adaptive_entropy_lambda}" \
    -e ADAPTIVE_ENTROPY_SCORE_MODE="${adaptive_entropy_score_mode}" \
    -e ADAPTIVE_ENTROPY_SIGNAL_MODE="${adaptive_entropy_signal_mode}" \
    -e ADAPTIVE_ENTROPY_SIGNAL_LAYER_IDX="${adaptive_entropy_signal_layer_idx}" \
    -e ADAPTIVE_MAX_RAW_START_SCALE_INDEX="${adaptive_max_raw_start_scale_index}" \
    -e STRUCTURE_LAYER_RATIO="${structure_layer_ratio}" \
    -e SVD_SOURCE_SCALE_INDEX="${svd_source_scale_index}" \
    -e TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL="${toprovar_entropy_side_channel_all}" \
    -e TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG="${toprovar_entropy_side_channel_debug}" \
    -e TOPROVAR_LAYER_RATIO_SCHEDULE="${TOPROVAR_LAYER_RATIO_SCHEDULE:-9:0.6,10:0.6}" \
    -e TOPROVAR_LOW_ENTROPY_PRUNE_RATIO="${toprovar_low_entropy_prune_ratio}" \
    -e TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO="${toprovar_high_entropy_prune_ratio}" \
    -e TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE="${toprovar_token_high_entropy_fraction_schedule}" \
    -e GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS:-}" \
    "${CONTAINER_NAME}" \
    bash -lc '
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate dpg

export PYTHONPATH="${INFINITY_DIR_IN_CONTAINER}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR_IN_CONTAINER}"
cd "${INFINITY_DIR_IN_CONTAINER}"

if [[ -z "${TOPROVAR_LOW_ENTROPY_PRUNE_RATIO}" ]]; then
    unset TOPROVAR_LOW_ENTROPY_PRUNE_RATIO
fi
if [[ -z "${TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO}" ]]; then
    unset TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO
fi
if [[ -z "${TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE}" ]]; then
    unset TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE
fi
if [[ "${TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL}" == "0" ]]; then
    unset TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL
fi
if [[ "${TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG}" == "0" ]]; then
    unset TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG
fi

extra_args=(
    --start_scale_index 9
    --direct_skip_start_scale_index 11
    --use_svd_depart_layers
    --structure_layer_ration "${STRUCTURE_LAYER_RATIO}"
    --svd_source_scale_index "${SVD_SOURCE_SCALE_INDEX}"
    --use_token_pruning
)
if [[ "${USE_ADAPTIVE_SCALE}" == "1" ]]; then
    extra_args+=(
        --use_adaptive_scale
        --adaptive_entropy_collect_start_scale_index "${ADAPTIVE_ENTROPY_COLLECT_START_SCALE_INDEX}"
        --adaptive_entropy_trigger_start_scale_index "${ADAPTIVE_ENTROPY_TRIGGER_START_SCALE_INDEX}"
        --adaptive_entropy_gamma "${ADAPTIVE_ENTROPY_GAMMA}"
        --adaptive_entropy_lambda "${ADAPTIVE_ENTROPY_LAMBDA}"
        --adaptive_entropy_score_mode "${ADAPTIVE_ENTROPY_SCORE_MODE}"
        --adaptive_entropy_signal_mode "${ADAPTIVE_ENTROPY_SIGNAL_MODE}"
        --adaptive_entropy_signal_layer_idx "${ADAPTIVE_ENTROPY_SIGNAL_LAYER_IDX}"
        --adaptive_max_raw_start_scale_index "${ADAPTIVE_MAX_RAW_START_SCALE_INDEX}"
    )
fi
gen_extra_args=()
if [[ -n "${GEN_EXTRA_ARGS:-}" ]]; then
    read -r -a gen_extra_args <<< "${GEN_EXTRA_ARGS}"
fi

python evaluation/gen_eval/infer4eval.py \
    --model_type infinity_8b \
    --model_path /data/models/infinity-8b \
    --checkpoint_type torch_shard \
    --vae_type 14 \
    --vae_path /data/models/infinity-2b/infinity_vae_d56_f8_14_patchify.pth \
    --pn 1M \
    --metadata_file "${METADATA_FILE_IN_CONTAINER}" \
    --outdir "${OUT_DIR_IN_CONTAINER}" \
    --text_encoder_ckpt /data/models/flan-t5-xl \
    --text_channels 2048 \
    --apply_spatial_patchify 1 \
    --n_samples "${N_SAMPLES}" \
    --bf16 1 \
    --seed "${SEED}" \
    --cfg 4 \
    --tau 1 \
    --rewrite_prompt 0 \
    --add_lvl_embeding_only_first_block 1 \
    --use_bit_label 1 \
    --rope2d_each_sa_layer 1 \
    --rope2d_normalized_by_hw 2 \
    "${extra_args[@]}" \
    "${gen_extra_args[@]}"
'

echo "GenEval generation completed: ${out_dir_host}"
