import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import cv2
import numpy as np
from tools.run_infinity import *
from cleanfid import fid
from tqdm import tqdm
import clip

from PIL import Image

import json


def main(args, data_type):
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    cfg = 4
    tau = 1.0
    h_div_w = 1/1 # aspect ratio, height:width
    seed = 1
    enable_positive_prompt=0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]


    with open(args.meta_data_json_path) as f:
        meta_data = json.load(f)

    save_root_dir = args.save_dir
    os.makedirs(save_root_dir,exist_ok=True)

    num_samples = 0
    max_samples = 800
    for img_id,data in tqdm(meta_data.items()):
        prompt = data['prompt']
        category = data['category']
        if data_type not in category:
            continue
        with torch.inference_mode():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=enable_positive_prompt,
            )
            os.makedirs(os.path.join(save_root_dir, category), exist_ok=True)
            cv2.imwrite(os.path.join(save_root_dir, category, f"{img_id}.png"), generated_image.cpu().numpy())

        num_samples += 1
        if num_samples >= max_samples:
            print(f"Processed {max_samples} samples; stopping generation.")
            break

    save_res_dir = f'{save_root_dir}/res'
    os.makedirs(save_res_dir,exist_ok=True)

    ref_dir = os.path.join(args.mjhq_reference_root, data_type)
    gen_dir = save_root_dir
    fid_score = fid.compute_fid(ref_dir,gen_dir)

    fid_content = f'FID score:{fid_score}'
    fid_save_path = f'{save_res_dir}/{data_type}_fid.txt'
    with open(fid_save_path, "w", encoding="utf-8") as f:
        f.write(fid_content)
    print(fid_content)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    def compute_clip_score(image_path, text):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([text],truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        return similarity

    generated_dir = save_root_dir
    total_score = 0
    count = 0

    for root, _, files in os.walk(generated_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure you're processing image files
                image_path = os.path.join(root, file)
                image_id = os.path.splitext(file)[0]
                if image_id in meta_data:
                    prompt = meta_data[image_id]["prompt"]
                    score = compute_clip_score(image_path, prompt)
                    total_score += score
                    count += 1
                else:
                    print(f"No prompt found for image {image_id}")

    if count > 0:
        average_clip_score = total_score / count

        clip_content = f"Average CLIP Score: {average_clip_score}"
        clip_save_path = f'{save_res_dir}/{data_type}_clip.txt'
        with open(clip_save_path, "w", encoding="utf-8") as f:
            f.write(clip_content)
        print(clip_content)
    else:
        print("No images were processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--use_8b', type=int, default=0)
    parser.add_argument('--meta_data_json', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--mjhq_reference_root', type=str, default=os.environ.get('MJHQ_REFERENCE_ROOT', ''))

    tmp_args = parser.parse_args()
    for required_path_arg in ('text_encoder_ckpt', 'vae_path', 'meta_data_json', 'save_dir', 'mjhq_reference_root'):
        if not getattr(tmp_args, required_path_arg):
            parser.error(f'--{required_path_arg} is required for MJHQ30K evaluation')

    text_encoder_ckpt = tmp_args.text_encoder_ckpt
    use_8b = (tmp_args.use_8b == 1)
    if use_8b:
        model_type='infinity_8b'
        checkpoint_type='torch_shard'
        vae_type=14
        apply_spatial_patchify=1
    else:
        model_type='infinity_2b'
        checkpoint_type='torch'
        vae_type=32
        apply_spatial_patchify=0

    cargs_dict = {
        'pn': '1M',  # 1M, 0.60M, 0.25M, 0.06M
        'model_path': tmp_args.model_path,
        'cfg_insertion_layer': 0,
        'vae_type': vae_type,
        'vae_path': tmp_args.vae_path,
        'add_lvl_embeding_only_first_block': 1,
        'use_bit_label': 1,
        'model_type': model_type,
        'rope2d_each_sa_layer': 1,
        'rope2d_normalized_by_hw': 2,
        'use_scale_schedule_embedding': 0,
        'sampling_per_bits': 1,
        'text_encoder_ckpt': text_encoder_ckpt,
        'text_channels': 2048,
        'apply_spatial_patchify': apply_spatial_patchify,
        'h_div_w_template': 1.000,
        'use_flex_attn': 0,
        'cache_dir': '/dev/shm',
        'checkpoint_type': checkpoint_type,
        'seed': 0,
        'bf16': 1,
        'save_file': 'tmp.jpg',
        'meta_data_json_path': tmp_args.meta_data_json,
        'save_dir': tmp_args.save_dir,
        'mjhq_reference_root': tmp_args.mjhq_reference_root,
        'enable_model_cache': 0,
    }

    tmp_dict = vars(tmp_args)

    for key, value in tmp_dict.items():
        if key not in cargs_dict:
            cargs_dict[key] = value

    cargs = argparse.Namespace(**cargs_dict)

    data_types = ['landscape', 'people','food']
    main(cargs, data_types[0])
