#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from infinity.models.fastvar_basic import FastVARCrossAttnBlock
from infinity.models.fastvar_utils import do_nothing
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.bench_toprovar_speed import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    build_scale_schedule,
    load_prompt_items,
    parse_cfg_list,
    parse_tau_list,
)
from tools.run_infinity import (
    add_common_arguments,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    parse_protected_global_layers,
)

DETAIL_LAYER_SOURCE_SCALE_INDEX = 8
DETAIL_LAYER_RATIO = 0.6
DETAIL_LAYER_SPLIT_MODE = "ratio"


def natural_key(path_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path_name))
    ]


def load_prompt_file_list(prompt_dir, prompt_files=None, prompt_manifest=None, prompt_limit=None):
    if prompt_files:
        files = list(prompt_files)
    elif prompt_manifest:
        files = []
        for raw_line in Path(prompt_manifest).read_text(encoding="utf-8").splitlines():
            line = re.sub(r"\s*#.*$", "", raw_line).strip()
            if line:
                files.append(line)
    else:
        prompt_root = Path(prompt_dir)
        if not prompt_root.is_dir():
            raise ValueError(f"prompt_dir does not exist: {prompt_dir}")
        files = [
            path.name
            for path in sorted(prompt_root.glob("*.txt"), key=lambda path: natural_key(path.name))
        ]

    if prompt_limit is not None and int(prompt_limit) > 0:
        files = files[: int(prompt_limit)]
    if not files:
        raise ValueError("No prompt files selected for adaptive scale SEGR-SSIM probe.")
    return files


def sanitize_prompt_stem(prompt_file, used_stems):
    stem = Path(prompt_file).stem
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._")
    if not stem:
        stem = "prompt"
    candidate = stem
    suffix = 1
    while candidate in used_stems:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used_stems.add(candidate)
    return candidate


def tensor_sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    meta = f"{array.dtype}:{array.shape}".encode("utf-8")
    digest = hashlib.sha256()
    digest.update(meta)
    digest.update(array.tobytes())
    return digest.hexdigest()


def image_sha256(rgb_uint8_hwc):
    if rgb_uint8_hwc.dtype != torch.uint8:
        raise ValueError("image_sha256 expects uint8 image tensor")
    digest = hashlib.sha256()
    digest.update(f"{tuple(rgb_uint8_hwc.shape)}".encode("utf-8"))
    digest.update(rgb_uint8_hwc.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_rgb_png(rgb_uint8_hwc, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8_hwc.detach().cpu().numpy(), mode="RGB").save(output_path)


def decode_codes_to_rgb_uint8(vae, summed_codes):
    decoded = vae.decode(summed_codes.squeeze(-3))
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp_(0.0, 1.0)
    decoded = decoded.mul(255.0).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).contiguous()
    return decoded


def build_gaussian_window(channels, window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    kernel_1d = torch.exp(-(coords.pow(2)) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim_rgb_uint8(image_a, image_b, window_size=11, sigma=1.5):
    if tuple(image_a.shape) != tuple(image_b.shape):
        raise ValueError(f"SSIM shape mismatch: {tuple(image_a.shape)} vs {tuple(image_b.shape)}")

    x = image_a.detach().cpu().float().permute(2, 0, 1).unsqueeze(0) / 255.0
    y = image_b.detach().cpu().float().permute(2, 0, 1).unsqueeze(0) / 255.0
    channels = int(x.shape[1])
    window = build_gaussian_window(channels, window_size=window_size, sigma=sigma, device=x.device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean().item())


@contextmanager
def patched_side_channel_probe(
    controller,
    *,
    enable_detail_layer_probe=False,
    svd_source_scale_index=DETAIL_LAYER_SOURCE_SCALE_INDEX,
    structure_layer_ration=DETAIL_LAYER_RATIO,
    svd_split_mode=DETAIL_LAYER_SPLIT_MODE,
):
    original_compute_merge = controller.compute_merge_from_entropy
    original_should_collect = controller.should_collect_attention_entropy
    original_should_record = controller.should_record_adaptive_entropy
    original_use_svd_depart_layers = bool(getattr(controller, "use_svd_depart_layers", False))
    original_structure_layer_ration = float(getattr(controller, "structure_layer_ration", 1.0))
    original_use_token_pruning = bool(getattr(controller, "use_token_pruning", False))
    original_svd_source_scale_ind = int(getattr(controller, "svd_source_scale_ind", svd_source_scale_index))
    original_svd_source_scale_index = int(getattr(controller, "svd_source_scale_index", svd_source_scale_index))
    original_svd_split_mode = getattr(controller, "svd_split_mode", svd_split_mode)

    def compute_merge_noop(self, x, attn_entropy_fn, x_shape, layer_idx, scale_ind):
        return do_nothing, do_nothing, do_nothing

    def should_collect_all(self, scale_ind):
        return True

    def should_record_all(self, scale_ind):
        return True

    controller.compute_merge_from_entropy = MethodType(compute_merge_noop, controller)
    controller.should_collect_attention_entropy = MethodType(should_collect_all, controller)
    controller.should_record_adaptive_entropy = MethodType(should_record_all, controller)
    if enable_detail_layer_probe:
        controller.use_svd_depart_layers = True
        controller.structure_layer_ration = float(structure_layer_ration)
        controller.use_token_pruning = False
        controller.svd_source_scale_ind = int(svd_source_scale_index)
        controller.svd_source_scale_index = int(svd_source_scale_index)
        controller.svd_split_mode = str(svd_split_mode)
    try:
        yield
    finally:
        controller.compute_merge_from_entropy = original_compute_merge
        controller.should_collect_attention_entropy = original_should_collect
        controller.should_record_adaptive_entropy = original_should_record
        controller.use_svd_depart_layers = original_use_svd_depart_layers
        controller.structure_layer_ration = original_structure_layer_ration
        controller.use_token_pruning = original_use_token_pruning
        controller.svd_source_scale_ind = original_svd_source_scale_ind
        controller.svd_source_scale_index = original_svd_source_scale_index
        controller.svd_split_mode = original_svd_split_mode


def build_detail_layer_metadata(controller, *, source_scale_index, structure_layer_ration):
    original_structure_layer_ration = float(getattr(controller, "structure_layer_ration", structure_layer_ration))
    try:
        controller.structure_layer_ration = float(structure_layer_ration)
        layer_sets, effective_ratio = controller._build_ratio_svd_layer_sets_for_scale(scale_ind=int(source_scale_index))
    finally:
        controller.structure_layer_ration = original_structure_layer_ration

    if layer_sets is None:
        raise RuntimeError(
            "failed to materialize detail/global layer sets from source-scale SVD scores"
        )

    controller._materialize_svd_layer_values()
    svd_layer_rho = {
        int(layer_idx): float(rho)
        for layer_idx, rho in sorted((controller.svd_layer_rho or {}).items())
    }
    num_layers = len(svd_layer_rho)
    detail_layers = sorted(int(x) for x in layer_sets["detail_layers"])
    global_layers = sorted(int(x) for x in layer_sets["global_layers"])
    return {
        "svd_source_scale_index": int(source_scale_index),
        "structure_layer_ration": float(structure_layer_ration),
        "svd_split_mode": DETAIL_LAYER_SPLIT_MODE,
        "svd_detail_layers": detail_layers,
        "svd_global_layers": global_layers,
        "svd_detail_layer_count": len(detail_layers),
        "svd_global_layer_count": len(global_layers),
        "num_layers": int(num_layers),
        "effective_detail_ratio": None if effective_ratio is None else float(effective_ratio),
        "svd_layer_rho": {str(layer_idx): rho for layer_idx, rho in svd_layer_rho.items()},
    }


@torch.no_grad()
def run_scale_probe(
    model,
    vae,
    text_cond_tuple,
    scale_schedule,
    cfg_list,
    tau_list,
    cfg_insertion_layers,
    seed,
    vae_type,
    apply_spatial_patchify,
    sampling_per_bits,
    collect_entropy,
    capture_intermediate_images,
):
    model.clear_cache()
    model.attnController.set_scale_schedule(scale_schedule)

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    batch_size = len(lens)
    if batch_size != 1:
        raise ValueError(f"probe expects single-prompt batch, got {batch_size}")

    if seed is None:
        rng = None
    else:
        model.rng.manual_seed(seed)
        rng = model.rng

    if apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2 * B
        kv_compact_un = kv_compact.clone()
        total = 0
        for length in lens:
            kv_compact_un[total : total + length] = model.cfg_uncond[:length]
            total += length
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
    else:
        bs = B

    kv_compact = model.text_norm(kv_compact)
    sos = cond_BD = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = model.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + model.pos_start.expand(bs, 1, -1)

    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = model.shared_ada_lin(cond_BD.float()).float().contiguous()

    abs_cfg_insertion_layers = []
    add_cfg_on_logits = False
    num_layers = len(model.unregistered_blocks)
    for item in cfg_insertion_layers:
        if item == 0:
            add_cfg_on_logits = True
        elif item == 1:
            raise NotImplementedError("cfg on probs is not supported in this probe path")
        elif item < 0:
            if num_layers + item <= 0:
                raise ValueError(f"invalid cfg_insertion_layer={item} for {num_layers} layers")
            abs_cfg_insertion_layers.append(num_layers + item)
        else:
            raise ValueError(f"invalid cfg_insertion_layer={item}")

    num_stages_minus_1 = len(scale_schedule) - 1
    summed_codes = 0
    scale_records = []
    layer_entropy_records = []
    head_entropy_records = []

    for block in model.unregistered_blocks:
        (block.sa if isinstance(block, FastVARCrossAttnBlock) else block.attn).kv_caching(True)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for scale_idx, pn in enumerate(scale_schedule):
            cfg = cfg_list[scale_idx]
            need_to_pad = 0
            attn_fn = None
            if model.use_flex_attn:
                attn_fn = model.attn_fn_compile_dict.get(tuple(scale_schedule[: (scale_idx + 1)]), None)

            layer_idx = 0
            for block_idx, block_chunk in enumerate(model.block_chunks):
                if model.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = model.add_lvl_embeding(last_stage, scale_idx, scale_schedule, need_to_pad=need_to_pad)
                if not model.add_lvl_embeding_only_first_block:
                    last_stage = model.add_lvl_embeding(last_stage, scale_idx, scale_schedule, need_to_pad=need_to_pad)

                for module in block_chunk.module:
                    last_stage = module(
                        x=last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=None,
                        attn_fn=attn_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=model.rope2d_freqs_grid,
                        scale_ind=scale_idx,
                        layer_idx=layer_idx,
                        x_shape=pn,
                        file_path=None,
                    )
                    if collect_entropy and isinstance(module, FastVARCrossAttnBlock):
                        layer_entropy = module.sa.get_latest_attn_entropy(scale_ind=scale_idx)
                        head_entropy = module.sa.get_latest_attn_entropy_per_head(scale_ind=scale_idx)
                        layer_entropy_records.append(
                            {
                                "scale_idx": int(scale_idx),
                                "scale_human": int(scale_idx + 1),
                                "scale_tokens": int(np.prod(pn)),
                                "scale_h": int(pn[1]),
                                "scale_w": int(pn[2]),
                                "layer_idx": int(layer_idx),
                                "layer_h_mean": float(layer_entropy.float().mean().item()),
                                "entropy_token_count": int(layer_entropy.numel()),
                            }
                        )
                        if head_entropy.ndim != 2:
                            raise RuntimeError(
                                f"expected per-head entropy shape [heads, tokens], got {tuple(head_entropy.shape)}"
                            )
                        head_entropy_mean = head_entropy.float().mean(dim=1)
                        for head_idx, head_h_mean in enumerate(head_entropy_mean):
                            head_entropy_records.append(
                                {
                                    "scale_idx": int(scale_idx),
                                    "scale_human": int(scale_idx + 1),
                                    "scale_tokens": int(np.prod(pn)),
                                    "scale_h": int(pn[1]),
                                    "scale_w": int(pn[2]),
                                    "layer_idx": int(layer_idx),
                                    "head_idx": int(head_idx),
                                    "head_h_mean": float(head_h_mean.item()),
                                    "entropy_token_count": int(head_entropy.shape[1]),
                                }
                            )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

            if (cfg != 1) and add_cfg_on_logits:
                logits_blv = model.get_logits(last_stage, cond_BD).mul(1 / tau_list[scale_idx])
                logits_blv = cfg * logits_blv[:B] + (1 - cfg) * logits_blv[B:]
            else:
                logits_blv = model.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[scale_idx])

            if not model.use_bit_label:
                raise NotImplementedError("probe only supports use_bit_label=1")

            tmp_bs, tmp_seq_len = logits_blv.shape[:2]
            logits_blv = logits_blv.reshape(tmp_bs, -1, 2)
            idx_bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_blv,
                rng=rng,
                top_k=DEFAULT_TOP_K,
                top_p=DEFAULT_TOP_P,
                num_samples=1,
            )[:, :, 0]
            idx_bld = idx_bld.reshape(tmp_bs, tmp_seq_len, -1)

            if vae_type == 0:
                raise NotImplementedError("probe only supports VAE decode path")

            assert pn[0] == 1
            idx_bld = idx_bld.reshape(B, pn[1], pn[2], -1)
            if apply_spatial_patchify:
                idx_bld = idx_bld.permute(0, 3, 1, 2)
                idx_bld = torch.nn.functional.pixel_shuffle(idx_bld, 2)
                idx_bld = idx_bld.permute(0, 2, 3, 1)
            idx_bld = idx_bld.unsqueeze(1)

            codes = vae.quantizer.lfq.indices_to_codes(idx_bld, label_type="bit_label")
            if scale_idx != num_stages_minus_1:
                summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[scale_idx + 1], mode=vae.quantizer.z_interplote_up)
                last_stage = last_stage.squeeze(-3)
                if apply_spatial_patchify:
                    last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
                last_stage = torch.permute(last_stage, [0, 2, 1])
            else:
                summed_codes += codes

            stats = None
            h_mean = None
            entropy_token_count = None
            entropy_layer_count = None
            if collect_entropy:
                stats = model.attnController.adaptive_entropy_stats_by_scale.get(int(scale_idx))
                if stats is None or int(stats.get("token_count", 0)) <= 0:
                    raise RuntimeError(f"missing entropy stats for scale {scale_idx}")
                entropy_sum_tensor = stats.get("entropy_sum_tensor")
                if entropy_sum_tensor is None:
                    raise RuntimeError(f"missing entropy_sum_tensor for scale {scale_idx}")
                entropy_token_count = int(stats["token_count"])
                entropy_layer_count = int(stats["layer_count"])
                h_mean = float(entropy_sum_tensor.item()) / float(entropy_token_count)

            scale_image = None
            if capture_intermediate_images:
                scale_image = decode_codes_to_rgb_uint8(vae, summed_codes)[0].cpu()

            scale_records.append(
                {
                    "scale_idx": int(scale_idx),
                    "scale_tokens": int(np.prod(pn)),
                    "scale_h": int(pn[1]),
                    "scale_w": int(pn[2]),
                    "h_mean": h_mean,
                    "entropy_token_count": entropy_token_count,
                    "entropy_layer_count": entropy_layer_count,
                    "token_hash": tensor_sha256(idx_bld),
                    "image": scale_image,
                }
            )

            if scale_idx != num_stages_minus_1:
                last_stage = model.word_embed(model.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs // B, 1, 1)

        final_image = decode_codes_to_rgb_uint8(vae, summed_codes)[0].cpu()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    finally:
        for block in model.unregistered_blocks:
            (block.sa if isinstance(block, FastVARCrossAttnBlock) else block.attn).kv_caching(False)

    return {
        "elapsed_ms": float(elapsed_ms),
        "final_image": final_image,
        "final_image_hash": image_sha256(final_image),
        "final_code_hash": tensor_sha256(summed_codes),
        "scale_records": scale_records,
        "layer_entropy_records": layer_entropy_records,
        "head_entropy_records": head_entropy_records,
    }


def build_probe_args(args):
    probe_args = deepcopy(args)
    probe_args.start_scale_index = 100
    probe_args.direct_skip_start_scale_index = 100
    probe_args.only_scale_skip = False
    probe_args.use_adaptive_scale = False
    probe_args.use_svd_depart_layers = False
    probe_args.use_token_pruning = False
    probe_args.scale_prop = 1.0
    probe_args.structure_layer_ration = 1.0
    probe_args.entropy_boundary_line = 1.0
    probe_args.svd_rho_threshold = -1.0
    probe_args.svd_source_scale_index = 8
    probe_args.svd_split_mode = "ratio"
    probe_args.protected_global_layers = []
    return probe_args


def resolve_cost_n_from_record(record):
    if record is None:
        return None
    h = int(record["scale_h"])
    w = int(record["scale_w"])
    if h <= 0 or w <= 0:
        return None
    if h == w:
        return float(h)
    return float((h * w) ** 0.5)


def build_entropy_score_trace(
    scale_records,
    epsilon,
    trigger_start_scale_index,
    max_raw_start_scale_index,
    adaptive_entropy_lambda=1.0,
    adaptive_entropy_score_mode="delta_h_width_power",
):
    prev_h = None
    adaptive_entropy_score_mode = str(adaptive_entropy_score_mode or "delta_h_width_power").strip().lower()
    for record_idx, record in enumerate(scale_records):
        h_mean = record["h_mean"]
        next_record = scale_records[record_idx + 1] if record_idx + 1 < len(scale_records) else None
        if h_mean is None or prev_h is None:
            record["delta_h"] = None
            record["cost_scale_index"] = None
            record["cost_n"] = None
            record["cost_value"] = None
            record["delta_cost_ratio"] = None
            record["adaptive_score_n"] = None
            record["adaptive_score"] = None
            record["adaptive_entropy_lambda"] = float(adaptive_entropy_lambda)
            record["adaptive_entropy_score_mode"] = adaptive_entropy_score_mode
            record["delta_rel"] = None
        else:
            delta_h = abs(float(h_mean) - float(prev_h))
            cost_scale_index = int(record["scale_idx"]) + 1
            cost_n = resolve_cost_n_from_record(next_record)
            cost_value = None if cost_n is None else float(cost_n) ** 4
            delta_cost_ratio = None if not cost_value else float(delta_h) / float(cost_value)
            adaptive_score_n = None if cost_n is None else float(cost_n)
            if adaptive_score_n is None:
                adaptive_score = None
            elif adaptive_entropy_score_mode == "legacy_delta_cost_ratio":
                adaptive_score = delta_cost_ratio
            else:
                adaptive_score = float(delta_h) * (adaptive_score_n ** float(adaptive_entropy_lambda))
            record["delta_h"] = delta_h
            record["cost_scale_index"] = cost_scale_index
            record["cost_n"] = cost_n
            record["cost_value"] = cost_value
            record["delta_cost_ratio"] = delta_cost_ratio
            record["adaptive_score_n"] = adaptive_score_n
            record["adaptive_score"] = adaptive_score
            record["adaptive_entropy_lambda"] = float(adaptive_entropy_lambda)
            record["adaptive_entropy_score_mode"] = adaptive_entropy_score_mode
            record["delta_rel"] = (float(h_mean) - float(prev_h)) / (float(prev_h) + float(epsilon))
        record["trigger_eligible"] = bool(
            int(record["scale_idx"]) >= int(trigger_start_scale_index)
            and int(record["scale_idx"]) <= int(max_raw_start_scale_index) - 1
        )
        if h_mean is not None:
            prev_h = float(h_mean)


def summarize_validation(prompt_reports, strict_validation):
    exact_final_image_matches = 0
    exact_final_code_matches = 0
    exact_all_scale_token_matches = 0
    min_final_ssim = None
    max_pixel_diff = 0

    for prompt_report in prompt_reports:
        validation = prompt_report["validation"]
        if validation["final_image_hash_match"]:
            exact_final_image_matches += 1
        if validation["final_code_hash_match"]:
            exact_final_code_matches += 1
        if validation["all_scale_token_hash_match"]:
            exact_all_scale_token_matches += 1

        final_ssim = float(validation["final_image_ssim"])
        min_final_ssim = final_ssim if min_final_ssim is None else min(min_final_ssim, final_ssim)
        max_pixel_diff = max(max_pixel_diff, int(validation["final_max_pixel_diff"]))

    num_prompts = len(prompt_reports)
    passed = (
        exact_final_image_matches == num_prompts
        or ((min_final_ssim or 0.0) >= 0.9999 and max_pixel_diff >= 0)
    )
    if strict_validation and not passed:
        raise RuntimeError(
            "side-channel consistency validation failed: "
            f"exact_final_image_matches={exact_final_image_matches}/{num_prompts}, "
            f"min_final_ssim={min_final_ssim}, max_pixel_diff={max_pixel_diff}"
        )

    return {
        "num_prompts": num_prompts,
        "exact_final_image_matches": exact_final_image_matches,
        "exact_final_code_matches": exact_final_code_matches,
        "exact_all_scale_token_matches": exact_all_scale_token_matches,
        "min_final_ssim": min_final_ssim,
        "max_pixel_diff": max_pixel_diff,
        "passed": bool(passed),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--prompt_files", nargs="*", default=None)
    parser.add_argument("--prompt_manifest", type=str, default="")
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--strict_validation", action="store_true")
    args = parser.parse_args()
    args.protected_global_layers = parse_protected_global_layers(args.protected_global_layers)
    return args


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL", "1")
    os.environ.setdefault("TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG", "1")

    args = parse_args()
    probe_args = build_probe_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"
    metrics_csv_path = output_dir / "metrics_per_scale.csv"
    metrics_per_layer_csv_path = output_dir / "metrics_per_layer.csv"
    metrics_per_head_csv_path = output_dir / "metrics_per_head.csv"
    report_json_path = output_dir / "report.json"
    run_config_path = output_dir / "run_config.yaml"

    prompt_files = load_prompt_file_list(
        prompt_dir=probe_args.prompt_dir,
        prompt_files=args.prompt_files,
        prompt_manifest=args.prompt_manifest,
        prompt_limit=args.prompt_limit if args.prompt_limit > 0 else None,
    )
    scale_schedule = build_scale_schedule(probe_args)
    cfg_list = parse_cfg_list(probe_args.cfg, len(scale_schedule))
    tau_list = parse_tau_list(probe_args.tau, len(scale_schedule))

    run_config = {
        "probe_name": "adaptive_scale_segr_ssim",
        "prompt_dir": probe_args.prompt_dir,
        "prompt_manifest": args.prompt_manifest or None,
        "prompt_limit": None if args.prompt_limit <= 0 else int(args.prompt_limit),
        "prompt_files": prompt_files,
        "output_dir": str(output_dir),
        "strict_validation": bool(args.strict_validation),
        "model": {
            "model_type": probe_args.model_type,
            "model_path": probe_args.model_path,
            "checkpoint_type": probe_args.checkpoint_type,
            "vae_type": int(probe_args.vae_type),
            "vae_path": probe_args.vae_path,
            "pn": probe_args.pn,
            "seed": int(probe_args.seed),
            "cfg": probe_args.cfg,
            "tau": probe_args.tau,
            "bf16": int(probe_args.bf16),
        },
        "probe_constraints": {
            "only_scale_skip": False,
            "use_token_pruning": False,
            "direct_skip_start_scale_index": 100,
            "compute_merge_from_entropy": "no_op",
            "collect_entropy_scales": "all",
            "main_forward_output": "plain_flash_attn_only",
            "entropy_output_usage": "side_channel_only",
            "side_channel_env": {
                "TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL": os.environ.get("TOPROVAR_ENTROPY_SIDE_CHANNEL_ALL"),
                "TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG": os.environ.get("TOPROVAR_ADAPTIVE_ENTROPY_SIDE_CHANNEL_DEBUG"),
            },
        },
        "detail_layer_probe": {
            "enabled_in_probe_run_only": True,
            "svd_source_scale_index": DETAIL_LAYER_SOURCE_SCALE_INDEX,
            "structure_layer_ration": DETAIL_LAYER_RATIO,
            "svd_split_mode": DETAIL_LAYER_SPLIT_MODE,
            "definition": "detail/global layers are split once from source scale 8 SVD scores at ratio 0.6 and then reused for analysis",
        },
        "head_entropy_probe": {
            "enabled_in_probe_run_only": True,
            "definition": "per-head entropy is read from flash-attn side-channel before averaging heads; it is analysis-only and does not affect logits, sampling, pruning, or images",
            "output_csv": str(metrics_per_head_csv_path.name),
        },
        "scale_schedule": [
            {
                "scale_idx": int(scale_idx),
                "scale_human": int(scale_idx + 1),
                "tokens": int(np.prod(pn)),
                "h": int(pn[1]),
                "w": int(pn[2]),
            }
            for scale_idx, pn in enumerate(scale_schedule)
        ],
        "ssim_reference": "plain-output final image with same prompt and same seed",
        "adaptive_score_config": {
            "collect_start_scale_index": int(probe_args.adaptive_entropy_collect_start_scale_index),
            "trigger_start_scale_index": int(probe_args.adaptive_entropy_trigger_start_scale_index),
            "max_raw_start_scale_index": int(probe_args.adaptive_max_raw_start_scale_index),
            "lambda": float(probe_args.adaptive_entropy_lambda),
            "score_mode": probe_args.adaptive_entropy_score_mode,
            "score_formula": "abs(delta_h) * candidate_width ** lambda",
            "cost_function": "n_next^4",
        },
        "delta_rel_epsilon": float(probe_args.adaptive_entropy_epsilon),
    }
    run_config_path.write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    text_tokenizer, text_encoder = load_tokenizer(t5_path=probe_args.text_encoder_ckpt)
    prompt_items = load_prompt_items(
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        prompt_dir=probe_args.prompt_dir,
        prompt_files=prompt_files,
    )

    vae = load_visual_tokenizer(probe_args)
    model = load_transformer(vae, probe_args)
    model.eval()

    used_stems = set()
    prompt_reports = []
    csv_rows = []
    layer_csv_rows = []
    head_csv_rows = []

    for prompt_index, item in enumerate(prompt_items):
        prompt_stem = sanitize_prompt_stem(item["name"], used_stems)
        prompt_image_dir = images_root / prompt_stem
        print(f"[probe] prompt {prompt_index + 1}/{len(prompt_items)}: {item['name']}", flush=True)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            plain_run = run_scale_probe(
                model=model,
                vae=vae,
                text_cond_tuple=item["text_cond_tuple"],
                scale_schedule=scale_schedule,
                cfg_list=cfg_list,
                tau_list=tau_list,
                cfg_insertion_layers=[probe_args.cfg_insertion_layer],
                seed=probe_args.seed,
                vae_type=probe_args.vae_type,
                apply_spatial_patchify=bool(probe_args.apply_spatial_patchify),
                sampling_per_bits=probe_args.sampling_per_bits,
                collect_entropy=False,
                capture_intermediate_images=False,
            )

        detail_layer_metadata = None
        with patched_side_channel_probe(
            model.attnController,
            enable_detail_layer_probe=True,
            svd_source_scale_index=DETAIL_LAYER_SOURCE_SCALE_INDEX,
            structure_layer_ration=DETAIL_LAYER_RATIO,
            svd_split_mode=DETAIL_LAYER_SPLIT_MODE,
        ):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                probe_run = run_scale_probe(
                    model=model,
                    vae=vae,
                    text_cond_tuple=item["text_cond_tuple"],
                    scale_schedule=scale_schedule,
                    cfg_list=cfg_list,
                    tau_list=tau_list,
                    cfg_insertion_layers=[probe_args.cfg_insertion_layer],
                    seed=probe_args.seed,
                    vae_type=probe_args.vae_type,
                    apply_spatial_patchify=bool(probe_args.apply_spatial_patchify),
                    sampling_per_bits=probe_args.sampling_per_bits,
                    collect_entropy=True,
                    capture_intermediate_images=True,
                )
            detail_layer_metadata = build_detail_layer_metadata(
                model.attnController,
                source_scale_index=DETAIL_LAYER_SOURCE_SCALE_INDEX,
                structure_layer_ration=DETAIL_LAYER_RATIO,
            )

        build_entropy_score_trace(
            probe_run["scale_records"],
            epsilon=probe_args.adaptive_entropy_epsilon,
            trigger_start_scale_index=probe_args.adaptive_entropy_trigger_start_scale_index,
            max_raw_start_scale_index=probe_args.adaptive_max_raw_start_scale_index,
            adaptive_entropy_lambda=probe_args.adaptive_entropy_lambda,
            adaptive_entropy_score_mode=probe_args.adaptive_entropy_score_mode,
        )
        if detail_layer_metadata is None:
            raise RuntimeError(f"missing detail layer metadata for prompt {item['name']}")

        plain_scale_hashes = [record["token_hash"] for record in plain_run["scale_records"]]
        probe_scale_hashes = [record["token_hash"] for record in probe_run["scale_records"]]
        final_pixel_diff = (
            plain_run["final_image"].to(torch.int16) - probe_run["final_image"].to(torch.int16)
        ).abs()
        final_ssim = compute_ssim_rgb_uint8(probe_run["final_image"], plain_run["final_image"])

        scale_metrics = []
        for scale_record in probe_run["scale_records"]:
            scale_idx = int(scale_record["scale_idx"])
            image_path = prompt_image_dir / f"scale_{scale_idx:02d}.png"
            save_rgb_png(scale_record["image"], image_path)
            ssim_to_final = compute_ssim_rgb_uint8(scale_record["image"], plain_run["final_image"])

            row = {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "seed": int(probe_args.seed),
                "scale_idx": scale_idx,
                "scale_human": scale_idx + 1,
                "scale_tokens": int(scale_record["scale_tokens"]),
                "scale_h": int(scale_record["scale_h"]),
                "scale_w": int(scale_record["scale_w"]),
                "h_mean": None if scale_record["h_mean"] is None else float(scale_record["h_mean"]),
                "delta_h": None if scale_record["delta_h"] is None else float(scale_record["delta_h"]),
                "cost_scale_index": scale_record["cost_scale_index"],
                "cost_n": None if scale_record["cost_n"] is None else float(scale_record["cost_n"]),
                "cost_value": None if scale_record["cost_value"] is None else float(scale_record["cost_value"]),
                "delta_cost_ratio": None if scale_record["delta_cost_ratio"] is None else float(scale_record["delta_cost_ratio"]),
                "adaptive_score_n": None if scale_record["adaptive_score_n"] is None else float(scale_record["adaptive_score_n"]),
                "adaptive_score": None if scale_record["adaptive_score"] is None else float(scale_record["adaptive_score"]),
                "adaptive_entropy_lambda": float(scale_record["adaptive_entropy_lambda"]),
                "adaptive_entropy_score_mode": scale_record["adaptive_entropy_score_mode"],
                "trigger_eligible": bool(scale_record["trigger_eligible"]),
                "delta_rel": None if scale_record["delta_rel"] is None else float(scale_record["delta_rel"]),
                "entropy_token_count": None if scale_record["entropy_token_count"] is None else int(scale_record["entropy_token_count"]),
                "entropy_layer_count": None if scale_record["entropy_layer_count"] is None else int(scale_record["entropy_layer_count"]),
                "ssim_to_final": float(ssim_to_final),
                "image_path": str(image_path.relative_to(output_dir)),
            }
            csv_rows.append(row)
            scale_metrics.append(row)

        layer_records_by_scale = {}
        for layer_record in probe_run["layer_entropy_records"]:
            scale_idx = int(layer_record["scale_idx"])
            layer_records_by_scale.setdefault(scale_idx, []).append(layer_record)
            layer_row = {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "seed": int(probe_args.seed),
                "scale_idx": scale_idx,
                "scale_human": int(layer_record["scale_human"]),
                "scale_tokens": int(layer_record["scale_tokens"]),
                "scale_h": int(layer_record["scale_h"]),
                "scale_w": int(layer_record["scale_w"]),
                "layer_idx": int(layer_record["layer_idx"]),
                "layer_h_mean": float(layer_record["layer_h_mean"]),
                "entropy_token_count": int(layer_record["entropy_token_count"]),
            }
            layer_csv_rows.append(layer_row)

        head_records_by_scale_layer = {}
        for head_record in probe_run["head_entropy_records"]:
            scale_idx = int(head_record["scale_idx"])
            layer_idx = int(head_record["layer_idx"])
            head_records_by_scale_layer.setdefault((scale_idx, layer_idx), []).append(head_record)
            head_row = {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "seed": int(probe_args.seed),
                "scale_idx": scale_idx,
                "scale_human": int(head_record["scale_human"]),
                "scale_tokens": int(head_record["scale_tokens"]),
                "scale_h": int(head_record["scale_h"]),
                "scale_w": int(head_record["scale_w"]),
                "layer_idx": layer_idx,
                "head_idx": int(head_record["head_idx"]),
                "head_h_mean": float(head_record["head_h_mean"]),
                "entropy_token_count": int(head_record["entropy_token_count"]),
            }
            head_csv_rows.append(head_row)

        for scale_record in probe_run["scale_records"]:
            scale_idx = int(scale_record["scale_idx"])
            expected_layer_count = int(scale_record["entropy_layer_count"] or 0)
            actual_layer_count = len(layer_records_by_scale.get(scale_idx, []))
            if expected_layer_count != actual_layer_count:
                raise RuntimeError(
                    f"per-layer entropy count mismatch for prompt {item['name']} scale {scale_idx}: "
                    f"expected {expected_layer_count}, got {actual_layer_count}"
                )
            for layer_record in layer_records_by_scale.get(scale_idx, []):
                layer_idx = int(layer_record["layer_idx"])
                actual_head_count = len(head_records_by_scale_layer.get((scale_idx, layer_idx), []))
                if actual_head_count <= 0:
                    raise RuntimeError(
                        f"missing per-head entropy for prompt {item['name']} scale {scale_idx} layer {layer_idx}"
                    )

        prompt_reports.append(
            {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "prompt_stem": prompt_stem,
                "text_len": int(item["text_len"]),
                "svd_source_scale_index": int(detail_layer_metadata["svd_source_scale_index"]),
                "structure_layer_ration": float(detail_layer_metadata["structure_layer_ration"]),
                "svd_split_mode": detail_layer_metadata["svd_split_mode"],
                "svd_detail_layers": detail_layer_metadata["svd_detail_layers"],
                "svd_global_layers": detail_layer_metadata["svd_global_layers"],
                "svd_detail_layer_count": int(detail_layer_metadata["svd_detail_layer_count"]),
                "svd_global_layer_count": int(detail_layer_metadata["svd_global_layer_count"]),
                "num_layers": int(detail_layer_metadata["num_layers"]),
                "effective_detail_ratio": detail_layer_metadata["effective_detail_ratio"],
                "svd_layer_rho": detail_layer_metadata["svd_layer_rho"],
                "plain_final": {
                    "elapsed_ms": float(plain_run["elapsed_ms"]),
                    "final_image_hash": plain_run["final_image_hash"],
                    "final_code_hash": plain_run["final_code_hash"],
                    "scale_token_hashes": plain_scale_hashes,
                },
                "probe": {
                    "elapsed_ms": float(probe_run["elapsed_ms"]),
                    "final_image_hash": probe_run["final_image_hash"],
                    "final_code_hash": probe_run["final_code_hash"],
                    "scale_token_hashes": probe_scale_hashes,
                    "image_dir": str(prompt_image_dir.relative_to(output_dir)),
                },
                "validation": {
                    "final_image_hash_match": plain_run["final_image_hash"] == probe_run["final_image_hash"],
                    "final_code_hash_match": plain_run["final_code_hash"] == probe_run["final_code_hash"],
                    "all_scale_token_hash_match": plain_scale_hashes == probe_scale_hashes,
                    "mismatched_scale_token_indices": [
                        idx
                        for idx, (lhs, rhs) in enumerate(zip(plain_scale_hashes, probe_scale_hashes))
                        if lhs != rhs
                    ],
                    "final_image_ssim": float(final_ssim),
                    "final_max_pixel_diff": int(final_pixel_diff.max().item()),
                },
                "scale_metrics": scale_metrics,
                "layer_entropy_records": probe_run["layer_entropy_records"],
                "head_entropy_record_count": len(probe_run["head_entropy_records"]),
            }
        )

        model.clear_cache()
        torch.cuda.empty_cache()

    expected_rows = len(prompt_items) * len(scale_schedule)
    if len(csv_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} CSV rows, got {len(csv_rows)}")

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_file",
                "prompt_text",
                "seed",
                "scale_idx",
                "scale_human",
                "scale_tokens",
                "scale_h",
                "scale_w",
                "h_mean",
                "delta_h",
                "cost_scale_index",
                "cost_n",
                "cost_value",
                "delta_cost_ratio",
                "adaptive_score_n",
                "adaptive_score",
                "adaptive_entropy_lambda",
                "adaptive_entropy_score_mode",
                "trigger_eligible",
                "delta_rel",
                "entropy_token_count",
                "entropy_layer_count",
                "ssim_to_final",
                "image_path",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(
                {
                    **row,
                    "h_mean": "" if row["h_mean"] is None else row["h_mean"],
                    "delta_h": "" if row["delta_h"] is None else row["delta_h"],
                    "cost_scale_index": "" if row["cost_scale_index"] is None else row["cost_scale_index"],
                    "cost_n": "" if row["cost_n"] is None else row["cost_n"],
                    "cost_value": "" if row["cost_value"] is None else row["cost_value"],
                    "delta_cost_ratio": "" if row["delta_cost_ratio"] is None else row["delta_cost_ratio"],
                    "adaptive_score_n": "" if row["adaptive_score_n"] is None else row["adaptive_score_n"],
                    "adaptive_score": "" if row["adaptive_score"] is None else row["adaptive_score"],
                    "trigger_eligible": int(bool(row["trigger_eligible"])),
                    "delta_rel": "" if row["delta_rel"] is None else row["delta_rel"],
                    "entropy_token_count": "" if row["entropy_token_count"] is None else row["entropy_token_count"],
                    "entropy_layer_count": "" if row["entropy_layer_count"] is None else row["entropy_layer_count"],
                }
            )

    with metrics_per_layer_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_file",
                "prompt_text",
                "seed",
                "scale_idx",
                "scale_human",
                "scale_tokens",
                "scale_h",
                "scale_w",
                "layer_idx",
                "layer_h_mean",
                "entropy_token_count",
            ],
        )
        writer.writeheader()
        for row in layer_csv_rows:
            writer.writerow(row)

    with metrics_per_head_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_file",
                "prompt_text",
                "seed",
                "scale_idx",
                "scale_human",
                "scale_tokens",
                "scale_h",
                "scale_w",
                "layer_idx",
                "head_idx",
                "head_h_mean",
                "entropy_token_count",
            ],
        )
        writer.writeheader()
        for row in head_csv_rows:
            writer.writerow(row)

    validation_summary = summarize_validation(prompt_reports, strict_validation=bool(args.strict_validation))
    report = {
        "run_config_path": str(run_config_path.relative_to(output_dir)),
        "metrics_per_scale_csv": str(metrics_csv_path.relative_to(output_dir)),
        "metrics_per_layer_csv": str(metrics_per_layer_csv_path.relative_to(output_dir)),
        "metrics_per_head_csv": str(metrics_per_head_csv_path.relative_to(output_dir)),
        "num_prompts": len(prompt_items),
        "num_scales": len(scale_schedule),
        "expected_rows": expected_rows,
        "actual_rows": len(csv_rows),
        "expected_layer_rows": len(prompt_items) * len(scale_schedule) * len(model.unregistered_blocks),
        "actual_layer_rows": len(layer_csv_rows),
        "actual_head_rows": len(head_csv_rows),
        "validation": {
            "side_channel_consistency": validation_summary,
            "criteria": {
                "preferred_exact_final_image_match": True,
                "fallback_min_final_ssim": 0.9999,
                "smoke_last_scale_min_ssim_to_final": 0.999,
            },
        },
        "prompt_reports": prompt_reports,
    }
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    del model
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[probe] wrote run config: {run_config_path}", flush=True)
    print(f"[probe] wrote metrics CSV: {metrics_csv_path}", flush=True)
    print(f"[probe] wrote per-layer metrics CSV: {metrics_per_layer_csv_path}", flush=True)
    print(f"[probe] wrote per-head metrics CSV: {metrics_per_head_csv_path}", flush=True)
    print(f"[probe] wrote report JSON: {report_json_path}", flush=True)
    print(
        json.dumps(
            {
                "num_prompts": len(prompt_items),
                "num_scales": len(scale_schedule),
                "rows": len(csv_rows),
                "layer_rows": len(layer_csv_rows),
                "head_rows": len(head_csv_rows),
                "validation": validation_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
