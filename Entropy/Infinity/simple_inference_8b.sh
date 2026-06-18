#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_ROOT="${TOPROVAR_MODEL_ROOT:-/data/models}"

python_ext="python"

model_type='infinity_8b'
model_path="${MODEL_ROOT}/infinity-8b"
checkpoint_type='torch_shard'
vae_type=14
vae_path="${MODEL_ROOT}/infinity-2b/infinity_vae_d56_f8_14_patchify.pth"
text_encoder_ckpt="${MODEL_ROOT}/flan-t5-xl"
apply_spatial_patchify=1
pn="1M"
out_dir="${SCRIPT_DIR}/outputs"
save_file="${out_dir}/compare_infinity_8b_cat_seed42.png"
prompt="A small fluffy kitten sitting calmly in the center of the frame, clean minimalist studio background, soft natural light, pale cream backdrop, no clutter, no extra objects, gentle shadows, bright eyes, detailed fur texture, photorealistic, sharp focus, elegant composition"

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
    --start_scale_index 100 \
    --save_file "${save_file}" \
    --prompt "${prompt}"

echo "推理完成，请检查输出文件: ${save_file}"
