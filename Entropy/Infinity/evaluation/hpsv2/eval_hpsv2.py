import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import sys

import cv2
import hpsv2
import torch
import numpy as np
from pytorch_lightning import seed_everything
from tools.nvtx_profiler import nvtx_profiler as nvtx

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_HOME = os.environ.get('HF_HOME', '/cache-dataset')


import random

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    
    args = parser.parse_args()
    if args.rewrite_prompt:
        parser.error("--rewrite_prompt is unsupported because tools.prompt_rewriter.py is not included in this release.")

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    seed_everything(args.seed)

    all_prompts = hpsv2.benchmark_prompts("all")

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to("cuda")
    elif args.model_type == 'flux_1_dev':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        
        
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

    total = 0
    for style, prompts in all_prompts.items():
        for idx, prompt in enumerate(prompts):
            total += 1
    ptr = 0
    for style, prompts in all_prompts.items():
        for idx, prompt in enumerate(prompts):
            ptr += 1
            if ptr % 10 == 0:
                print(f'Generate {ptr}/{total} images...')
            if ptr > 100:
                break
            
            image_save_file_path = os.path.join(args.outdir, style, f"{idx:05d}.jpg")
            os.makedirs(osp.dirname(image_save_file_path), exist_ok=True)

            tau = args.tau
            cfg = args.cfg
            images = []
            for _ in range(args.n_samples):
                t1 = time.time()
                if args.model_type == 'sdxl':
                    image = base(
                        prompt=prompt,
                        num_inference_steps=40,
                        denoising_end=0.8,
                        output_type="latent",
                    ).images
                    image = refiner(
                        prompt=prompt,
                        num_inference_steps=40,
                        denoising_start=0.8,
                        image=image,
                    ).images[0]
                elif args.model_type == 'sd3':
                    image = pipe(
                        prompt,
                        negative_prompt="",
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        num_images_per_prompt=1,
                    ).images[0]
                elif args.model_type == 'flux_1_dev':
                    image = pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        max_sequence_length=512,
                        num_images_per_prompt=1,
                    ).images[0]
                elif args.model_type == 'flux_1_dev_schnell':
                    image = pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=0.0,
                        num_inference_steps=4,
                        max_sequence_length=256,
                        generator=torch.Generator("cpu").manual_seed(0)
                    ).images[0]
                elif args.model_type == 'pixart_sigma':
                    image = pipe(prompt).images[0]
                elif 'infinity' in args.model_type:
                    file_path = os.path.join(args.outdir, f"{idx:05d}")
                    
                    h_div_w_template = 1.000
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                    with nvtx.range(f'Image generate {ptr}'):
                        image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type,  file_path=file_path)
                else:
                    raise ValueError
                t2 = time.time()
                print(f'{args.model_type} infer one image takes {t2-t1:.2f}s')
                images.append(image)
            
            for i, image in enumerate(images):
                if 'infinity' in args.model_type:
                    cv2.imwrite(image_save_file_path, image.cpu().numpy())
                else:
                    image.save(image_save_file_path)
    nvtx.print_stats()
    # hpsv2.evaluate(args.outdir, hps_version="v2.1")
