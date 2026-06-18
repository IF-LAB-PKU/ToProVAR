#!/bin/bash
set -euo pipefail

# Environment and paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# Reduce CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Select the CUDA device with CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_ROOT="${TOPROVAR_MODEL_ROOT:-/data/models}"

python_ext="python"

# 8B model settings with the ToProVAR switches preserved.
model_type='infinity_8b'
model_path="${MODEL_ROOT}/infinity-8b"
checkpoint_type='torch_shard'
vae_type=14
vae_path="${MODEL_ROOT}/infinity-2b/infinity_vae_d56_f8_14_patchify.pth"
text_encoder_ckpt="${MODEL_ROOT}/flan-t5-xl"
apply_spatial_patchify=1
pn="1M"
out_dir="${SCRIPT_DIR}/outputs"
save_file="${out_dir}/toprovar_infinity_8b_detail_seed42.png"
prompt="An ultra-detailed futuristic cyberpunk megacity, advanced sci-fi architecture, dense street-level details, sleek flying vehicles, holographic neon signs, glowing circuitry, reflective glass and metal surfaces, intricate mechanical structures, high-tech city skyline, sharp focus, crisp edges, realistic lighting, deep contrast, volumetric fog, rich textures, fine-grained details, highly detailed windows, cables, panels, and urban infrastructure"

mkdir -p "${out_dir}"

# Run inference with a single sample to reduce memory pressure.
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
    --start_scale_index 100 \
    --save_file "${save_file}" \
    --prompt "${prompt}"

echo "推理完成，请检查输出文件: ${save_file}"
