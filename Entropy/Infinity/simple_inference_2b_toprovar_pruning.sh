#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_ROOT="${TOPROVAR_MODEL_ROOT:-/data/models}"

python_ext="python"

# 2B model settings.
model_type='infinity_2b'
model_path="${MODEL_ROOT}/infinity-2b/infinity_2b_reg.pth"
checkpoint_type='torch'
vae_type=32
vae_path="${MODEL_ROOT}/infinity-2b/infinity_vae_d32reg.pth"
text_encoder_ckpt="${MODEL_ROOT}/flan-t5-xl"
apply_spatial_patchify=0

missing=0
for required_path in "${model_path}" "${vae_path}" "${text_encoder_ckpt}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required model path: ${required_path}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "Set TOPROVAR_MODEL_ROOT=/path/to/models or prepare the expected model files under ${MODEL_ROOT}." >&2
    exit 1
fi

# Resolution / output
pn="1M"
out_dir="${SCRIPT_DIR}/outputs"
save_file="${out_dir}/compare_infinity_2b_toprovar_pruning_seed42.png"

# Detail-sensitive prompt for qualitative comparison
prompt="A street-level cyberpunk avenue at night after rain, ultra-dense sci-fi city details, layered holographic billboards, reflective wet pavement, transparent shop windows packed with gadgets, cables, pipes, vents, metal panels, glowing control screens, parked delivery drones, passing flying vehicles, distant skyscrapers filled with tiny lit windows, sharp focus, realistic lighting, crisp edges, high contrast, atmospheric depth, volumetric fog, extremely detailed materials and urban infrastructure"

# ToProVAR pruning config:
# - keep only token pruning on
# - do not enable scale skipping
# - start only at the last scale to avoid reusing already-pruned entropy across later scales
start_scale_index=10

mkdir -p "${out_dir}"
cd "${SCRIPT_DIR}"

${python_ext} tools/run_infinity.py \
    --model_type ${model_type} \
    --model_path "${model_path}" \
    --checkpoint_type ${checkpoint_type} \
    --vae_type ${vae_type} \
    --vae_path "${vae_path}" \
    --pn ${pn} \
    --text_encoder_ckpt "${text_encoder_ckpt}" \
    --text_channels 2048 \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --outdir "${out_dir}" \
    --n_samples 1 \
    --bf16 1 \
    --seed 42 \
    --cfg 4 \
    --tau 1 \
    --add_lvl_embeding_only_first_block 1 \
    --use_bit_label 1 \
    --rope2d_each_sa_layer 1 \
    --rope2d_normalized_by_hw 2 \
    --start_scale_index ${start_scale_index} \
    --use_token_pruning \
    --save_file "${save_file}" \
    --prompt "${prompt}"

echo "推理完成，请检查输出文件: ${save_file}"
