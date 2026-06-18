#!/bin/bash



infer_eval_hpsv21() {
   
    mkdir -p ${out_dir}

    {  ${python_ext} evaluation/hpsv2/eval_hpsv2.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --n_samples 1 \
    --start_scale_index 100 \
    --only_scale_skip; } 2>&1  | tee ${out_dir}/log.txt 
}



cuda_device=3
target_mod=1  # default value; override with --mod

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            cuda_device="$2"
            shift 2
            ;;
        --mod)
            target_mod="$2"
            # Validate that --mod is one of 0, 1, or 2.
            if ! [[ "$target_mod" =~ ^[0-2]$ ]]; then
                echo "错误：--mod参数必须是0、1或2"
                exit 1
            fi
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFINITY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${INFINITY_DIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export CUDA_VISIBLE_DEVICES=$cuda_device
python_ext="uv run python"


# User-configurable parameters.

# Model selection: set to 1 for the 8B model, or 0 for the 2B model.
use_8b=0
n_samples=4
MODEL_ROOT="${TOPROVAR_MODEL_ROOT:-/data/models}"
EVAL_OUTPUT_ROOT="${TOPROVAR_EVAL_OUTPUT_ROOT:-${INFINITY_DIR}/outputs/evaluation}"


# Configure model paths and settings from use_8b.
if [ $use_8b -eq 1 ]; then
    model_type='infinity_8b'
    infinity_model_path="${MODEL_ROOT}/infinity-8b"
    checkpoint_type='torch_shard'
    vae_type=14
    vae_path="${MODEL_ROOT}/infinity-2b/infinity_vae_d56_f8_14_patchify.pth"
    apply_spatial_patchify=1
else
    model_type='infinity_2b'
    infinity_model_path="${MODEL_ROOT}/infinity-2b/infinity_2b_reg.pth"
    checkpoint_type='torch'
    vae_type=32
    vae_path="${MODEL_ROOT}/infinity-2b/infinity_vae_d32reg.pth"
    apply_spatial_patchify=0
fi

out_dir_home="${EVAL_OUTPUT_ROOT}/infinity_${model_type##*_}_evaluation"

# Inference parameters.
text_channels=2048
text_encoder_ckpt="${MODEL_ROOT}/flan-t5-xl"
use_bit_label=1
use_scale_schedule_embedding=0
rope2d_each_sa_layer=1
cfg_insertion_layer=0
add_lvl_embeding_only_first_block=1
rope2d_normalized_by_hw=2
tau=1
cfg=4
pn=1M

rewrite_prompt=0
    
out_dir_root=${out_dir_home}/naive

# Run HPS v2.1 evaluation.
out_dir=${out_dir_root}/hpsv21
infer_eval_hpsv21
