import os
import os.path as osp
import time
import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

INFINITY_ROOT = Path(__file__).resolve().parents[2]
if str(INFINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(INFINITY_ROOT))

from tools.run_infinity import add_common_arguments, dynamic_resolution_h_w, gen_one_img, load_tokenizer, load_transformer, load_visual_tokenizer

os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"


def tensor_to_image_array(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def save_image(image, save_file: str):
    if isinstance(image, torch.Tensor):
        cv2.imwrite(save_file, tensor_to_image_array(image))
    else:
        image.save(save_file)


def load_oracle_start_map(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {f"{idx // 4}:{idx % 4}": value for idx, value in enumerate(data)}
    raise ValueError(f"Unsupported oracle start map format: {path}")


def resolve_oracle_start(oracle_map, index: int, sample_idx: int, metadata: dict):
    if oracle_map is None:
        return None
    keys = [
        f"{index}:{sample_idx}",
        f"{index:05d}:{sample_idx}",
        str(index),
        f"{index:05d}",
        metadata.get("prompt", ""),
    ]
    for key in keys:
        if key not in oracle_map:
            continue
        value = oracle_map[key]
        if isinstance(value, dict):
            value = (
                value.get("effective_start_scale_index")
                or value.get("start_scale_index")
                or value.get("effective_start")
            )
        if value is not None:
            return int(value)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        "--legacy_geneval_seed",
        action="store_true",
        help="Match the older GenEval wrapper: seed once per prompt and do not pass g_seed per sample.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="First metadata index to generate, inclusive.")
    parser.add_argument("--end_index", type=int, default=-1, help="Last metadata index to generate, exclusive; -1 means all.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip prompts whose expected sample jpgs already exist.")
    
    args = parser.parse_args()
    print(f"legacy_geneval_seed={args.legacy_geneval_seed}")
    if args.rewrite_prompt:
        parser.error("--rewrite_prompt is unsupported because tools.prompt_rewriter.py is not included in this release.")
    if not args.metadata_file:
        parser.error("--metadata_file is required for GenEval inference.")

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    if args.model_type == "flux_1_dev":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == "flux_1_dev_schnell":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif "infinity" in args.model_type:
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        vae = load_visual_tokenizer(args)
        infinity = load_transformer(vae, args)
        scale_schedule = dynamic_resolution_h_w[1.000][args.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    oracle_start_map = load_oracle_start_map(getattr(args, "oracle_start_map", ""))

    for index, metadata in tqdm(enumerate(metadatas), total=len(metadatas)):
        if index < args.start_index:
            continue
        if args.end_index >= 0 and index >= args.end_index:
            continue
        print(args.outdir)
        seed_everything(args.seed)
        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata["prompt"]

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        expected_samples = [os.path.join(sample_path, f"{i:05}.jpg") for i in range(args.n_samples)]
        if args.skip_existing and all(os.path.exists(path) for path in expected_samples):
            print(f"[skip_existing] metadata={index} already has {args.n_samples} samples")
            continue
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        tau = args.tau
        cfg = args.cfg

        images = []
        
        for sample_j in range(args.n_samples):
            sample_seed = None
            if not args.legacy_geneval_seed:
                sample_seed = args.seed + sample_j
                seed_everything(sample_seed)
            t1 = time.time()
            if args.model_type == "flux_1_dev":
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == "flux_1_dev_schnell":
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.model_type == "pixart_sigma":
                image = pipe(prompt).images[0]
            elif "infinity" in args.model_type:
                oracle_start = resolve_oracle_start(oracle_start_map, index, sample_j, metadata)
                if oracle_start is not None:
                    infinity.attnController.set_fixed_start_scale_index(oracle_start)
                    print(f"[Oracle Start] metadata={index} sample={sample_j} start={oracle_start}")
                image = gen_one_img(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt,
                    tau_list=tau,
                    cfg_sc=3,
                    cfg_list=cfg,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    g_seed=None if args.legacy_geneval_seed else sample_seed,
                    file_path=args.outdir if args.legacy_geneval_seed else outpath,
                )
            else:
                raise ValueError
            t2 = time.time()
            print(f"{args.model_type} infer one image takes {t2-t1:.2f}s")
            images.append(image)

        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            save_image(image, save_file)
