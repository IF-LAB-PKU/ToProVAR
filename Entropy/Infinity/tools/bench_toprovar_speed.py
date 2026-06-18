#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from infinity.models.fastvar_basic import FastVARCrossAttnBlock
from infinity.models.fastvar_utils import (
    do_nothing,
    parse_toprovar_token_high_entropy_fraction_schedule,
    skip_layer,
)
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from tools.run_infinity import (
    add_common_arguments,
    encode_prompt,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    parse_protected_global_layers,
)


DEFAULT_TOP_K = 900
DEFAULT_TOP_P = 0.97


def parse_cfg_list(cfg_value, num_scales):
    if isinstance(cfg_value, list):
        cfg_list = [float(x) for x in cfg_value]
    else:
        cfg_list = [float(x) for x in str(cfg_value).split(",")]
    if len(cfg_list) == 1:
        cfg_list = cfg_list * num_scales
    if len(cfg_list) < num_scales:
        raise ValueError(f"cfg list too short: need {num_scales}, got {len(cfg_list)}")
    return cfg_list


def parse_tau_list(tau_value, num_scales):
    if isinstance(tau_value, list):
        tau_list = [float(x) for x in tau_value]
    else:
        tau_list = [float(x) for x in str(tau_value).split(",")]
    if len(tau_list) == 1:
        tau_list = tau_list * num_scales
    if len(tau_list) < num_scales:
        raise ValueError(f"tau list too short: need {num_scales}, got {len(tau_list)}")
    return tau_list


def build_scale_schedule(args):
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in scale_schedule]


def install_token_trace(model):
    controller = model.attnController
    if getattr(controller, "_bench_trace_installed", False):
        return

    original = controller.compute_merge_from_entropy

    def traced_compute_merge(self, x, attn_entropy_fn, x_shape, layer_idx, scale_ind):
        full_len = int(x.shape[1])
        key = (int(scale_ind), int(layer_idx))
        merge_fn, unmerge_fn, idx_fn = original(
            x=x,
            attn_entropy_fn=attn_entropy_fn,
            x_shape=x_shape,
            layer_idx=layer_idx,
            scale_ind=scale_ind,
        )

        prompt_trace = getattr(self, "_bench_trace", None)
        logged_keys = getattr(self, "_bench_logged_keys", None)
        if prompt_trace is None or logged_keys is None:
            return merge_fn, unmerge_fn, idx_fn

        if merge_fn is skip_layer:
            if key not in logged_keys:
                prompt_trace[key] = {
                    "scale_idx": int(scale_ind),
                    "layer_idx": int(layer_idx),
                    "full_len": full_len,
                    "kept_len": 0,
                    "mode": "skip",
                }
                logged_keys.add(key)
            return merge_fn, unmerge_fn, idx_fn

        if merge_fn is do_nothing:
            if key not in logged_keys:
                prompt_trace[key] = {
                    "scale_idx": int(scale_ind),
                    "layer_idx": int(layer_idx),
                    "full_len": full_len,
                    "kept_len": full_len,
                    "mode": "full",
                }
                logged_keys.add(key)
            return merge_fn, unmerge_fn, idx_fn

        def wrapped_merge(cur_x):
            out = merge_fn(cur_x)
            if key not in logged_keys:
                prompt_trace[key] = {
                    "scale_idx": int(scale_ind),
                    "layer_idx": int(layer_idx),
                    "full_len": int(cur_x.shape[1]),
                    "kept_len": int(out.shape[1]),
                    "mode": "prune",
                }
                logged_keys.add(key)
            return out

        return wrapped_merge, unmerge_fn, idx_fn

    controller.compute_merge_from_entropy = MethodType(traced_compute_merge, controller)
    controller._bench_trace_installed = True


def reset_token_trace(model):
    controller = model.attnController
    controller._bench_trace = {}
    controller._bench_logged_keys = set()


def collect_token_trace(model, expected_records):
    controller = model.attnController
    trace = getattr(controller, "_bench_trace", {})
    if len(trace) != expected_records:
        raise RuntimeError(f"expected {expected_records} token records, got {len(trace)}")
    return [trace[key] for key in sorted(trace.keys())]


def build_abs_cfg_insertion_layers(model, cfg_insertion_layers):
    abs_cfg_insertion_layers = []
    add_cfg_on_logits = False
    add_cfg_on_probs = False
    num_layers = len(model.unregistered_blocks)

    for item in cfg_insertion_layers:
        if item == 0:
            add_cfg_on_logits = True
        elif item == 1:
            add_cfg_on_probs = True
        elif item < 0:
            if num_layers + item <= 0:
                raise ValueError(f"invalid cfg_insertion_layer={item} for {num_layers} layers")
            abs_cfg_insertion_layers.append(num_layers + item)
        else:
            raise ValueError(f"invalid cfg_insertion_layer={item}")

    if add_cfg_on_probs:
        raise NotImplementedError("cfg on probs is not used in this benchmark path")
    return abs_cfg_insertion_layers, add_cfg_on_logits


@torch.no_grad()
def run_scale_loop_only(
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
):
    model.clear_cache()
    model.attnController.set_scale_schedule(scale_schedule)

    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    batch_size = len(lens)
    if batch_size != 1:
        raise ValueError(f"this benchmark expects a single prompt batch, got {batch_size}")

    if g_seed_is_none(seed):
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

    abs_cfg_insertion_layers, add_cfg_on_logits = build_abs_cfg_insertion_layers(model, cfg_insertion_layers)
    num_stages_minus_1 = len(scale_schedule) - 1
    summed_codes = 0

    for block in model.unregistered_blocks:
        (block.sa if isinstance(block, FastVARCrossAttnBlock) else block.attn).kv_caching(True)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for si, pn in enumerate(scale_schedule):
            cfg = cfg_list[si]
            need_to_pad = 0
            attn_fn = None
            if model.use_flex_attn:
                attn_fn = model.attn_fn_compile_dict.get(tuple(scale_schedule[: (si + 1)]), None)

            layer_idx = 0
            for block_idx, block_chunk in enumerate(model.block_chunks):
                if model.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = model.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not model.add_lvl_embeding_only_first_block:
                    last_stage = model.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)

                for module in block_chunk.module:
                    last_stage = module(
                        x=last_stage,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=None,
                        attn_fn=attn_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=model.rope2d_freqs_grid,
                        scale_ind=si,
                        layer_idx=layer_idx,
                        x_shape=pn,
                        file_path=None,
                    )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

            model.attnController.finalize_adaptive_scale(si)

            if (cfg != 1) and add_cfg_on_logits:
                logits_blv = model.get_logits(last_stage, cond_BD).mul(1 / tau_list[si])
                logits_blv = cfg * logits_blv[:B] + (1 - cfg) * logits_blv[B:]
            else:
                logits_blv = model.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])

            if model.use_bit_label:
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
            else:
                raise NotImplementedError("this benchmark only supports use_bit_label=1")

            if vae_type == 0:
                raise NotImplementedError("this benchmark only supports VAE decode path")

            assert pn[0] == 1
            idx_bld = idx_bld.reshape(B, pn[1], pn[2], -1)
            if apply_spatial_patchify:
                idx_bld = idx_bld.permute(0, 3, 1, 2)
                idx_bld = torch.nn.functional.pixel_shuffle(idx_bld, 2)
                idx_bld = idx_bld.permute(0, 2, 3, 1)
            idx_bld = idx_bld.unsqueeze(1)

            codes = vae.quantizer.lfq.indices_to_codes(idx_bld, label_type="bit_label")
            if si != num_stages_minus_1:
                summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up)
                last_stage = last_stage.squeeze(-3)
                if apply_spatial_patchify:
                    last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
                last_stage = torch.permute(last_stage, [0, 2, 1])
            else:
                summed_codes += codes

            if si != num_stages_minus_1:
                last_stage = model.word_embed(model.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs // B, 1, 1)

        img = vae.decode(summed_codes.squeeze(-3))
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    finally:
        for block in model.unregistered_blocks:
            (block.sa if isinstance(block, FastVARCrossAttnBlock) else block.attn).kv_caching(False)

    return elapsed_ms, img


def g_seed_is_none(seed):
    return seed is None


def compute_flops_from_trace(trace_records, scale_schedule, depth, embed_dim, hidden_dim, text_len, batch_size, kv_dim):
    trace_map = {(item["scale_idx"], item["layer_idx"]): item for item in trace_records}
    self_attn_flops = 0.0
    cross_attn_flops = 0.0
    ffn_flops = 0.0
    cumulative_k = {layer_idx: 0 for layer_idx in range(depth)}
    full_tokens_all_blocks = 0
    actual_tokens_all_blocks = 0
    skipped_blocks = 0

    for scale_idx, scale in enumerate(scale_schedule):
        full_len = int(np.prod(scale))
        for layer_idx in range(depth):
            full_tokens_all_blocks += full_len
            record = trace_map[(scale_idx, layer_idx)]
            kept_len = int(record["kept_len"])
            actual_tokens_all_blocks += kept_len

            if kept_len == 0:
                skipped_blocks += 1
                continue

            cumulative_k[layer_idx] += kept_len
            self_attn_flops += batch_size * (
                8.0 * kept_len * embed_dim * embed_dim
                + 4.0 * kept_len * cumulative_k[layer_idx] * embed_dim
            )
            cross_attn_flops += batch_size * (
                4.0 * kept_len * embed_dim * embed_dim
                + 4.0 * kept_len * text_len * embed_dim
                + 4.0 * text_len * kv_dim * embed_dim
            )
            ffn_flops += batch_size * (4.0 * kept_len * embed_dim * hidden_dim)

    total_flops = self_attn_flops + cross_attn_flops + ffn_flops
    actual_prune_ratio = 1.0 - (actual_tokens_all_blocks / full_tokens_all_blocks)
    return {
        "self_attn_flops": self_attn_flops,
        "cross_attn_flops": cross_attn_flops,
        "ffn_flops": ffn_flops,
        "total_flops": total_flops,
        "self_attn_tflops": self_attn_flops / 1e12,
        "cross_attn_tflops": cross_attn_flops / 1e12,
        "ffn_tflops": ffn_flops / 1e12,
        "total_tflops": total_flops / 1e12,
        "full_tokens_all_blocks": full_tokens_all_blocks,
        "actual_tokens_all_blocks": actual_tokens_all_blocks,
        "actual_prune_ratio_all_blocks": actual_prune_ratio,
        "skipped_blocks": skipped_blocks,
    }


def summarize_run(prompt_results):
    pure_ms = np.array([item["pure_infer_ms"] for item in prompt_results], dtype=np.float64)
    encode_ms = np.array([item["encode_ms"] for item in prompt_results], dtype=np.float64)
    total_tflops = np.array([item["flops"]["total_tflops"] for item in prompt_results], dtype=np.float64)
    actual_prune = np.array([item["flops"]["actual_prune_ratio_all_blocks"] for item in prompt_results], dtype=np.float64)
    tracked_ms = np.array([item["tracked_total_ms"] for item in prompt_results], dtype=np.float64)
    component_names = ("self_attn", "ffn", "cross_attn")
    mean_component_ms = {}
    for name in component_names:
        values = np.array([item["timing_ms"].get(name, 0.0) for item in prompt_results], dtype=np.float64)
        mean_component_ms[name] = float(values.mean())
    raw_start_hist = {}
    effective_start_hist = {}
    first_pruned_hist = {}
    for item in prompt_results:
        for hist, key in (
            (raw_start_hist, "raw_start_scale_index"),
            (effective_start_hist, "effective_start_scale_index"),
            (first_pruned_hist, "first_pruned_scale_index"),
        ):
            value = item.get(key)
            bucket = "none" if value is None else str(int(value))
            hist[bucket] = hist.get(bucket, 0) + 1
    return {
        "num_prompts": len(prompt_results),
        "mean_encode_ms": float(encode_ms.mean()),
        "median_encode_ms": float(np.median(encode_ms)),
        "mean_pure_infer_ms": float(pure_ms.mean()),
        "median_pure_infer_ms": float(np.median(pure_ms)),
        "min_pure_infer_ms": float(pure_ms.min()),
        "max_pure_infer_ms": float(pure_ms.max()),
        "mean_embed_to_image_ms": float((encode_ms + pure_ms).mean()),
        "median_embed_to_image_ms": float(np.median(encode_ms + pure_ms)),
        "mean_tracked_infer_ms": float(tracked_ms.mean()),
        "mean_component_ms": mean_component_ms,
        "mean_total_tflops": float(total_tflops.mean()),
        "median_total_tflops": float(np.median(total_tflops)),
        "mean_actual_prune_ratio_all_blocks": float(actual_prune.mean()),
        "raw_start_scale_hist": raw_start_hist,
        "effective_start_scale_hist": effective_start_hist,
        "first_pruned_scale_hist": first_pruned_hist,
    }


def run_benchmark_mode(mode_name, mode_args, prompt_items, scale_schedule):
    print(f"[bench] loading {mode_name} model", flush=True)
    vae = load_visual_tokenizer(mode_args)
    model = load_transformer(vae, mode_args)
    model.eval()
    install_token_trace(model)

    depth = len(model.unregistered_blocks)
    hidden_dim = round(model.C * model.mlp_ratio / 256) * 256
    expected_records = depth * len(scale_schedule)
    cfg_list = parse_cfg_list(mode_args.cfg, len(scale_schedule))
    tau_list = parse_tau_list(mode_args.tau, len(scale_schedule))

    warmup_item = prompt_items[0]
    reset_token_trace(model)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = run_scale_loop_only(
            model=model,
            vae=vae,
            text_cond_tuple=warmup_item["text_cond_tuple"],
            scale_schedule=scale_schedule,
            cfg_list=cfg_list,
            tau_list=tau_list,
            cfg_insertion_layers=[mode_args.cfg_insertion_layer],
            seed=mode_args.seed,
            vae_type=mode_args.vae_type,
            apply_spatial_patchify=bool(mode_args.apply_spatial_patchify),
            sampling_per_bits=mode_args.sampling_per_bits,
        )
    torch.cuda.synchronize()
    model.clear_cache()

    prompt_results = []
    for item in prompt_items:
        print(f"[bench] {mode_name} prompt={item['name']}", flush=True)
        reset_token_trace(model)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            elapsed_ms, _ = run_scale_loop_only(
                model=model,
                vae=vae,
                text_cond_tuple=item["text_cond_tuple"],
                scale_schedule=scale_schedule,
                cfg_list=cfg_list,
                tau_list=tau_list,
                cfg_insertion_layers=[mode_args.cfg_insertion_layer],
                seed=mode_args.seed,
                vae_type=mode_args.vae_type,
                apply_spatial_patchify=bool(mode_args.apply_spatial_patchify),
                sampling_per_bits=mode_args.sampling_per_bits,
            )

        trace_records = collect_token_trace(model, expected_records)
        flops = compute_flops_from_trace(
            trace_records=trace_records,
            scale_schedule=scale_schedule,
            depth=depth,
            embed_dim=model.C,
            hidden_dim=hidden_dim,
            text_len=item["text_len"],
            batch_size=2 if any(np.array(cfg_list) != 1) else 1,
            kv_dim=model.C,
        )
        controller = model.attnController
        timing_stats = {
            name: float(total_ms)
            for name, total_ms, _avg_ms, _count, _pct in controller.get_timing_stats()
        }
        tracked_total_ms = float(sum(timing_stats.values()))
        adaptive_segr = controller.get_adaptive_segr_report()
        snapshot_scale_index = adaptive_segr.get("effective_start_scale_index") or mode_args.start_scale_index
        start_scale_snapshot = controller.get_svd_snapshot_for_scale(snapshot_scale_index)
        prompt_results.append(
            {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "text_len": item["text_len"],
                "encode_ms": float(item["encode_ms"]),
                "pure_infer_ms": float(elapsed_ms),
                "flops": flops,
                "timing_ms": timing_stats,
                "tracked_total_ms": tracked_total_ms,
                "svd_pruning_layers": sorted(int(x) for x in start_scale_snapshot["pruning_layers"]),
                "svd_global_layers": sorted(int(x) for x in start_scale_snapshot["global_layers"]),
                "svd_detail_layers": sorted(int(x) for x in start_scale_snapshot["detail_layers"]),
                "effective_layer_detail_ratio_by_scale": controller.get_effective_layer_detail_ratio_by_scale_report(),
                "effective_token_prune_ratio_by_scale": controller.get_effective_token_prune_ratio_by_scale_report(),
                "effective_token_high_entropy_fraction_by_scale": controller.get_effective_token_high_entropy_fraction_by_scale_report(),
                "svd_pruning_layers_by_scale": controller.get_svd_pruning_layers_by_scale_report(),
                "svd_layer_rho": {
                    str(int(layer_idx)): float(rho)
                    for layer_idx, rho in sorted((controller.svd_layer_rho or {}).items())
                },
                "svd_split_method": getattr(controller, "svd_split_method", None),
                "svd_split_gap": None if getattr(controller, "svd_split_gap", None) is None else float(controller.svd_split_gap),
                "svd_effective_rho_threshold": None if getattr(controller, "svd_effective_rho_threshold", None) is None else float(controller.svd_effective_rho_threshold),
                "svd_gap_metric": getattr(controller, "svd_gap_metric", None),
                "svd_gap_value": None if getattr(controller, "svd_gap_value", None) is None else float(controller.svd_gap_value),
                "svd_detail_layer_count": len(start_scale_snapshot["detail_layers"]),
                "svd_global_layer_count": len(start_scale_snapshot["global_layers"]),
                "svd_fallback_reason": getattr(controller, "svd_fallback_reason", None),
                "svd_source_scale_index": int(getattr(controller, "svd_source_scale_ind", mode_args.svd_source_scale_index)),
                "raw_start_scale_index": adaptive_segr.get("raw_start_scale_index"),
                "effective_start_scale_index": adaptive_segr.get("effective_start_scale_index"),
                "first_pruned_scale_index": adaptive_segr.get("first_pruned_scale_index"),
                "adaptive_segr": adaptive_segr,
                "svd_split_mode": getattr(controller, "svd_split_mode", mode_args.svd_split_mode),
                "svd_rho_threshold": None if getattr(controller, "svd_rho_threshold", None) is None else float(controller.svd_rho_threshold),
                "protected_global_layers": sorted(int(x) for x in getattr(controller, "protected_global_layers", [])),
            }
        )
        model.clear_cache()

    summary = summarize_run(prompt_results)

    del model
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode_name,
        "summary": summary,
        "prompts": prompt_results,
    }


def load_prompt_items(text_tokenizer, text_encoder, prompt_dir, prompt_files):
    items = []
    for prompt_file in prompt_files:
        prompt_path = Path(prompt_dir) / prompt_file
        text = prompt_path.read_text(encoding="utf-8").strip()
        start_time = time.perf_counter()
        text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, text, enable_positive_prompt=False)
        encode_ms = (time.perf_counter() - start_time) * 1000.0
        items.append(
            {
                "name": prompt_file,
                "text": text,
                "text_len": int(text_cond_tuple[1][0]),
                "text_cond_tuple": text_cond_tuple,
                "encode_ms": float(encode_ms),
            }
        )
    return items


def build_baseline_args(args):
    baseline_args = deepcopy(args)
    baseline_args.start_scale_index = 100
    baseline_args.direct_skip_start_scale_index = 100
    baseline_args.only_scale_skip = False
    baseline_args.use_adaptive_scale = False
    baseline_args.use_svd_depart_layers = False
    baseline_args.use_token_pruning = False
    baseline_args.scale_prop = 1.0
    baseline_args.structure_layer_ration = 1.0
    baseline_args.entropy_boundary_line = 1.0
    baseline_args.svd_rho_threshold = -1.0
    baseline_args.svd_source_scale_index = 7
    baseline_args.svd_split_mode = 'ratio'
    baseline_args.protected_global_layers = parse_protected_global_layers('')
    return baseline_args


def build_speedup_report(baseline_result, prune_result):
    base_summary = baseline_result["summary"]
    prune_summary = prune_result["summary"]
    component_speedups = {}
    for name in ("self_attn", "ffn", "cross_attn"):
        base_value = float(base_summary["mean_component_ms"].get(name, 0.0))
        prune_value = float(prune_summary["mean_component_ms"].get(name, 0.0))
        component_speedups[name] = None if prune_value <= 0.0 else (base_value / prune_value)
    return {
        "embed_to_image_speedup": base_summary["mean_embed_to_image_ms"] / prune_summary["mean_embed_to_image_ms"],
        "pure_infer_speedup": base_summary["mean_pure_infer_ms"] / prune_summary["mean_pure_infer_ms"],
        "pure_flops_speedup": base_summary["mean_total_tflops"] / prune_summary["mean_total_tflops"],
        "pure_component_speedups": component_speedups,
        "mean_encode_ms": {
            "baseline": base_summary["mean_encode_ms"],
            "prune": prune_summary["mean_encode_ms"],
        },
        "mean_embed_to_image_ms": {
            "baseline": base_summary["mean_embed_to_image_ms"],
            "prune": prune_summary["mean_embed_to_image_ms"],
        },
        "mean_pure_infer_ms": {
            "baseline": base_summary["mean_pure_infer_ms"],
            "prune": prune_summary["mean_pure_infer_ms"],
        },
        "mean_component_ms": {
            "baseline": base_summary["mean_component_ms"],
            "prune": prune_summary["mean_component_ms"],
        },
        "mean_total_tflops": {
            "baseline": base_summary["mean_total_tflops"],
            "prune": prune_summary["mean_total_tflops"],
        },
        "mean_actual_prune_ratio_all_blocks": prune_summary["mean_actual_prune_ratio_all_blocks"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--prompt_files", nargs="*", default=None)
    parser.add_argument("--prompt_manifest", type=str, default="")
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--timing_stats", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.protected_global_layers = parse_protected_global_layers(args.protected_global_layers)
    return args


def natural_key(path_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path_name))
    ]


def load_prompt_file_list(prompt_dir, prompt_files=None, prompt_manifest=None, prompt_limit=None):
    if prompt_files:
        files = list(prompt_files)
    elif prompt_manifest:
        files = [
            line.strip()
            for line in Path(prompt_manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
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
        raise ValueError("No prompt files selected for speed benchmark.")
    return files


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    os.environ["TOPROVAR_ENABLE_TIMING_STATS"] = "1" if args.timing_stats else "0"
    prompt_files = load_prompt_file_list(
        prompt_dir=args.prompt_dir,
        prompt_files=args.prompt_files,
        prompt_manifest=args.prompt_manifest,
        prompt_limit=args.prompt_limit if args.prompt_limit > 0 else None,
    )
    scale_schedule = build_scale_schedule(args)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    prompt_items = load_prompt_items(
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        prompt_dir=args.prompt_dir,
        prompt_files=prompt_files,
    )

    baseline_args = build_baseline_args(args)
    baseline_result = run_benchmark_mode("baseline", baseline_args, prompt_items, scale_schedule)
    prune_result = run_benchmark_mode("current_prune", args, prompt_items, scale_schedule)
    speedup_report = build_speedup_report(baseline_result, prune_result)

    output = {
        "prompt_files": prompt_files,
        "scale_schedule_tokens": [int(np.prod(item)) for item in scale_schedule],
        "pruning_config": {
            "start_scale_index": int(args.start_scale_index),
            "direct_skip_start_scale_index": int(args.direct_skip_start_scale_index),
            "use_svd_depart_layers": bool(args.use_svd_depart_layers),
            "structure_layer_ration": float(args.structure_layer_ration),
            "use_token_pruning": bool(args.use_token_pruning),
            "svd_source_scale_index": int(args.svd_source_scale_index),
            "svd_split_mode": args.svd_split_mode,
            "svd_rho_threshold": None if args.svd_rho_threshold <= 0 else float(args.svd_rho_threshold),
            "protected_global_layers": parse_protected_global_layers(args.protected_global_layers),
            "use_adaptive_scale": bool(args.use_adaptive_scale),
            "adaptive_entropy_collect_start_scale_index": int(args.adaptive_entropy_collect_start_scale_index),
            "adaptive_entropy_trigger_start_scale_index": int(args.adaptive_entropy_trigger_start_scale_index),
            "adaptive_entropy_s_min": None if args.adaptive_entropy_s_min is None else int(args.adaptive_entropy_s_min),
            "adaptive_entropy_gamma": float(args.adaptive_entropy_gamma),
            "adaptive_entropy_epsilon": float(args.adaptive_entropy_epsilon),
            "adaptive_entropy_lambda": float(args.adaptive_entropy_lambda),
            "adaptive_entropy_score_mode": args.adaptive_entropy_score_mode,
            "adaptive_entropy_signal_mode": args.adaptive_entropy_signal_mode,
            "adaptive_entropy_signal_layer_idx": int(args.adaptive_entropy_signal_layer_idx),
            "adaptive_max_raw_start_scale_index": int(args.adaptive_max_raw_start_scale_index),
            "low_entropy_prune_ratio": None
            if os.environ.get("TOPROVAR_LOW_ENTROPY_PRUNE_RATIO") is None
            else float(os.environ["TOPROVAR_LOW_ENTROPY_PRUNE_RATIO"]),
            "high_entropy_prune_ratio": None
            if os.environ.get("TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO") is None
            else float(os.environ["TOPROVAR_HIGH_ENTROPY_PRUNE_RATIO"]),
            "token_ratio_schedule": os.environ.get("TOPROVAR_TOKEN_RATIO_SCHEDULE", ""),
            "token_high_entropy_fraction_schedule": {
                str(int(scale_ind)): float(fraction)
                for scale_ind, fraction in sorted(
                    parse_toprovar_token_high_entropy_fraction_schedule(
                        os.environ.get("TOPROVAR_TOKEN_HIGH_ENTROPY_FRACTION_SCHEDULE")
                    ).items()
                )
            },
        },
        "baseline": baseline_result,
        "current_prune": prune_result,
        "speedups": speedup_report,
        "notes": {
            "embed_to_image_boundary": "measured from immediately before prompt text encoding to immediately after final image tensor materialization",
            "pure_infer_boundary": "measured from immediately before the Scale 1 autoregressive loop to immediately after final image tensor materialization",
            "pure_component_boundary": "mean CUDA timing totals reported by the controller for self_attn, cross_attn, and ffn during the pure inference region",
            "flops_scope": "analytical transformer block FLOPs (self-attn, cross-attn, FFN) using prompt-specific kept-token counts; excludes control overhead like SVD, entropy interpolation, token selection, and VAE decode",
        },
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(speedup_report, indent=2), flush=True)
    print(f"[bench] report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
