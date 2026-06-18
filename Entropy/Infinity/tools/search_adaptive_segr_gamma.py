#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

import torch

from tools.bench_toprovar_speed import (
    build_scale_schedule,
    load_prompt_items,
    parse_cfg_list,
    parse_tau_list,
    run_scale_loop_only,
)
from tools.run_infinity import (
    add_common_arguments,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    parse_protected_global_layers,
)


DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SCORE_MODE = "delta_h_width_power"
TARGET_RAW_RATIOS = {8: 0.05, 9: 0.90, 10: 0.05}
TARGET_EFFECTIVE_RATIOS = {9: 0.95, 10: 0.05}


def parse_float_grid(raw_value, default_values=()):
    if raw_value is None or str(raw_value).strip() == "":
        return list(default_values)
    return [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def natural_key(path_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path_name))
    ]


def load_prompt_file_list(prompt_dir, prompt_files=None, prompt_manifest=None, prompt_limit=100):
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
        raise ValueError("No prompt files selected for adaptive lambda/gamma search.")
    return files


def histogram_count(hist, value):
    key = "none" if value is None else str(int(value))
    hist[key] = hist.get(key, 0) + 1


def get_hist_count(hist, scale_index):
    return int(hist.get(str(int(scale_index)), 0))


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def compute_adaptive_score(trace_item, lambda_value):
    delta_h = finite_float(trace_item.get("delta_h"))
    score_n = finite_float(trace_item.get("adaptive_score_n"))
    if score_n is None:
        score_n = finite_float(trace_item.get("cost_n"))
    if delta_h is None or score_n is None or score_n <= 0.0:
        return None
    return float(delta_h) * (float(score_n) ** float(lambda_value))


def trace_score_values(prompt_reports, lambda_value, trigger_start_scale_index, max_raw_start):
    values = []
    for prompt_report in prompt_reports:
        for item in prompt_report["trace"]:
            scale_idx = int(item["scale_idx"])
            if scale_idx < int(trigger_start_scale_index) or scale_idx > int(max_raw_start) - 1:
                continue
            score = compute_adaptive_score(item, lambda_value=lambda_value)
            if score is not None:
                values.append(float(score))
    return values


def interval_width(lower, upper):
    if lower is None or upper is None:
        return 0.0, 0.0
    width = max(float(upper) - float(lower), 0.0)
    if lower > 0.0 and upper > 0.0:
        return width, max(math.log(float(upper)) - math.log(float(lower)), 0.0)
    return width, 0.0


def build_gamma_candidates(score_values, extra_gammas=None):
    values = sorted({float(value) for value in score_values if math.isfinite(float(value))})
    if not values:
        raise ValueError("No finite adaptive scores were available for gamma search.")

    candidates = {}
    first = values[0]
    last = values[-1]
    below = first - max(abs(first) * 0.01, 1e-12)
    above = last + max(abs(last) * 0.01, 1e-12)
    candidates[below] = {"lower": None, "upper": first, "stability_width": 0.0, "stability_log_width": 0.0}
    candidates[above] = {"lower": last, "upper": None, "stability_width": 0.0, "stability_log_width": 0.0}

    for lower, upper in zip(values, values[1:]):
        gamma = math.sqrt(lower * upper) if lower > 0.0 and upper > 0.0 else (lower + upper) * 0.5
        width, log_width = interval_width(lower, upper)
        candidates[gamma] = {
            "lower": lower,
            "upper": upper,
            "stability_width": width,
            "stability_log_width": log_width,
        }

    for gamma in extra_gammas or []:
        gamma = float(gamma)
        if not math.isfinite(gamma):
            continue
        lower = None
        upper = None
        for value in values:
            if value < gamma:
                lower = value
            elif value > gamma:
                upper = value
                break
        width, log_width = interval_width(lower, upper)
        candidates[gamma] = {
            "lower": lower,
            "upper": upper,
            "stability_width": width,
            "stability_log_width": log_width,
        }

    return candidates


def replay_trace(trace, gamma, lambda_value, trigger_start_scale_index, max_raw_start, min_effective_start):
    raw_start = None
    effective_start = None
    replay_rows = []

    for item in sorted(trace, key=lambda row: int(row["scale_idx"])):
        scale_idx = int(item["scale_idx"])
        delta_h = finite_float(item.get("delta_h"))
        cost_scale_index = item.get("cost_scale_index")
        cost_n = finite_float(item.get("cost_n"))
        cost_value = finite_float(item.get("cost_value"))
        delta_cost_ratio = finite_float(item.get("delta_cost_ratio"))
        adaptive_score_n = finite_float(item.get("adaptive_score_n"))
        if adaptive_score_n is None:
            adaptive_score_n = cost_n
        adaptive_score = compute_adaptive_score(item, lambda_value=lambda_value)
        trigger_eligible = (
            scale_idx >= int(trigger_start_scale_index)
            and scale_idx <= int(max_raw_start) - 1
        )
        candidate_raw = scale_idx + 1 if trigger_eligible else None
        triggered = False

        if trigger_eligible and raw_start is None and adaptive_score is not None and adaptive_score > float(gamma):
            raw_start = min(candidate_raw, int(max_raw_start))
            effective_start = max(raw_start, int(min_effective_start))
            triggered = True

        if raw_start is None and scale_idx >= int(max_raw_start) - 1:
            raw_start = int(max_raw_start)
            effective_start = max(raw_start, int(min_effective_start))
            triggered = True

        replay_rows.append(
            {
                "scale_idx": scale_idx,
                "h_mean": item.get("h_mean"),
                "delta_h": delta_h,
                "cost_scale_index": cost_scale_index,
                "cost_n": cost_n,
                "cost_value": cost_value,
                "delta_cost_ratio": delta_cost_ratio,
                "adaptive_score_n": adaptive_score_n,
                "adaptive_score": adaptive_score,
                "adaptive_entropy_lambda": float(lambda_value),
                "adaptive_entropy_score_mode": SCORE_MODE,
                "trigger_eligible": bool(trigger_eligible),
                "candidate_raw_start_scale_index": candidate_raw,
                "triggered": triggered,
                "raw_start_scale_index": raw_start,
                "effective_start_scale_index": effective_start,
            }
        )
        if raw_start is not None:
            break

    return raw_start, effective_start, replay_rows


def summarize_score_bands(prompt_reports, lambda_values, target_scales=(7, 8, 9)):
    summary = {}
    for lambda_value in lambda_values:
        score_values_by_scale = {int(scale_idx): [] for scale_idx in target_scales}
        for prompt_report in prompt_reports:
            for item in prompt_report["trace"]:
                scale_idx = int(item["scale_idx"])
                if scale_idx not in score_values_by_scale:
                    continue
                score = compute_adaptive_score(item, lambda_value=lambda_value)
                if score is not None:
                    score_values_by_scale[scale_idx].append(float(score))

        stats = {}
        for scale_idx, values in score_values_by_scale.items():
            values = sorted(values)
            stats[str(int(scale_idx))] = {
                "count": len(values),
                "min": None if not values else float(values[0]),
                "max": None if not values else float(values[-1]),
                "mean": None if not values else float(sum(values) / len(values)),
            }
        summary[str(float(lambda_value))] = stats
    return summary


def histogram_error(hist, target_ratios, num_prompts):
    return sum(
        abs(get_hist_count(hist, scale_index) - float(ratio) * float(num_prompts))
        for scale_index, ratio in target_ratios.items()
    )


def build_search_reports(
    prompt_reports,
    lambda_values,
    extra_gammas,
    trigger_start_scale_index,
    max_raw_start,
    min_effective_start,
):
    search_reports = []
    num_prompts = len(prompt_reports)
    for lambda_value in lambda_values:
        values = trace_score_values(
            prompt_reports=prompt_reports,
            lambda_value=lambda_value,
            trigger_start_scale_index=trigger_start_scale_index,
            max_raw_start=max_raw_start,
        )
        gamma_candidates = build_gamma_candidates(values, extra_gammas=extra_gammas)
        for gamma, interval in sorted(gamma_candidates.items()):
            raw_hist = {}
            effective_hist = {}
            prompt_rows = []
            for prompt_report in prompt_reports:
                raw_start, effective_start, replay_rows = replay_trace(
                    prompt_report["trace"],
                    gamma=gamma,
                    lambda_value=lambda_value,
                    trigger_start_scale_index=trigger_start_scale_index,
                    max_raw_start=max_raw_start,
                    min_effective_start=min_effective_start,
                )
                histogram_count(raw_hist, raw_start)
                histogram_count(effective_hist, effective_start)
                prompt_rows.append(
                    {
                        "prompt_file": prompt_report["prompt_file"],
                        "raw_start_scale_index": raw_start,
                        "effective_start_scale_index": effective_start,
                    }
                )

            raw_error = histogram_error(raw_hist, TARGET_RAW_RATIOS, num_prompts)
            effective_error = histogram_error(effective_hist, TARGET_EFFECTIVE_RATIOS, num_prompts)
            search_reports.append(
                {
                    "lambda": float(lambda_value),
                    "gamma": float(gamma),
                    "adaptive_entropy_score_mode": SCORE_MODE,
                    "gamma_interval": interval,
                    "raw_start_hist": raw_hist,
                    "effective_start_hist": effective_hist,
                    "selection_metrics": {
                        "raw8_count": get_hist_count(raw_hist, 8),
                        "raw8_ratio": get_hist_count(raw_hist, 8) / max(num_prompts, 1),
                        "raw9_count": get_hist_count(raw_hist, 9),
                        "raw9_ratio": get_hist_count(raw_hist, 9) / max(num_prompts, 1),
                        "raw10_count": get_hist_count(raw_hist, 10),
                        "raw10_ratio": get_hist_count(raw_hist, 10) / max(num_prompts, 1),
                        "effective9_count": get_hist_count(effective_hist, 9),
                        "effective9_ratio": get_hist_count(effective_hist, 9) / max(num_prompts, 1),
                        "effective10_count": get_hist_count(effective_hist, 10),
                        "effective10_ratio": get_hist_count(effective_hist, 10) / max(num_prompts, 1),
                        "raw_histogram_l1_error": float(raw_error),
                        "effective_histogram_l1_error": float(effective_error),
                    },
                    "prompts": prompt_rows,
                }
            )
    return search_reports


def build_selected_prompt_replays(
    prompt_reports,
    gamma,
    lambda_value,
    trigger_start_scale_index,
    max_raw_start,
    min_effective_start,
):
    prompt_rows = []
    for prompt_report in prompt_reports:
        raw_start, effective_start, replay_rows = replay_trace(
            prompt_report["trace"],
            gamma=gamma,
            lambda_value=lambda_value,
            trigger_start_scale_index=trigger_start_scale_index,
            max_raw_start=max_raw_start,
            min_effective_start=min_effective_start,
        )
        prompt_rows.append(
            {
                "prompt_file": prompt_report["prompt_file"],
                "raw_start_scale_index": raw_start,
                "effective_start_scale_index": effective_start,
                "replay_trace": replay_rows,
            }
        )
    return prompt_rows


def select_preferred_config(search_reports):
    if not search_reports:
        raise ValueError("search_reports is empty")
    chosen = min(
        search_reports,
        key=lambda report: (
            report["selection_metrics"]["raw_histogram_l1_error"],
            report["selection_metrics"]["effective_histogram_l1_error"],
            -float(report["gamma_interval"].get("stability_log_width", 0.0)),
            -float(report["gamma_interval"].get("stability_width", 0.0)),
            abs(float(report["lambda"]) - 1.0),
            float(report["gamma"]),
        ),
    )
    return {
        "gamma": float(chosen["gamma"]),
        "lambda": float(chosen["lambda"]),
        "adaptive_entropy_score_mode": SCORE_MODE,
        "reason": "min_raw_error_then_effective_error_then_gamma_stability_then_lambda_near_one",
        "source": "lambda_gamma_replay_grid",
        "raw_start_hist": chosen["raw_start_hist"],
        "effective_start_hist": chosen["effective_start_hist"],
        "selection_metrics": chosen["selection_metrics"],
        "gamma_interval": chosen["gamma_interval"],
    }


def write_histogram_csv(path, search_reports, num_prompts):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lambda",
                "gamma",
                "histogram",
                "scale_index",
                "count",
                "percent",
                "raw_error",
                "effective_error",
                "stability_width",
                "stability_log_width",
            ],
        )
        writer.writeheader()
        for report in search_reports:
            for hist_name in ("raw_start_hist", "effective_start_hist"):
                for scale_index, count in sorted(report[hist_name].items(), key=lambda item: natural_key(item[0])):
                    writer.writerow(
                        {
                            "lambda": report["lambda"],
                            "gamma": report["gamma"],
                            "histogram": hist_name,
                            "scale_index": scale_index,
                            "count": count,
                            "percent": count / max(num_prompts, 1),
                            "raw_error": report["selection_metrics"]["raw_histogram_l1_error"],
                            "effective_error": report["selection_metrics"]["effective_histogram_l1_error"],
                            "stability_width": report["gamma_interval"].get("stability_width", 0.0),
                            "stability_log_width": report["gamma_interval"].get("stability_log_width", 0.0),
                        }
                    )


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--prompt_files", nargs="*", default=None)
    parser.add_argument("--prompt_manifest", type=str, default="")
    parser.add_argument("--prompt_limit", type=int, default=100)
    parser.add_argument("--lambda_grid", type=str, default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument("--gamma_grid", type=str, default="")
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--report_csv", type=str, default="")
    args = parser.parse_args()
    args.protected_global_layers = parse_protected_global_layers(args.protected_global_layers)
    return args


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    prompt_files = load_prompt_file_list(
        prompt_dir=args.prompt_dir,
        prompt_files=args.prompt_files,
        prompt_manifest=args.prompt_manifest,
        prompt_limit=args.prompt_limit,
    )
    lambda_grid = sorted(set(parse_float_grid(args.lambda_grid, default_values=DEFAULT_LAMBDAS)))
    extra_gamma_grid = sorted(set(parse_float_grid(args.gamma_grid, default_values=())))

    probe_args = deepcopy(args)
    probe_args.use_adaptive_scale = True
    probe_args.svd_source_scale_index = 8
    probe_args.adaptive_entropy_gamma = 1e30
    probe_args.adaptive_entropy_lambda = 1.0
    probe_args.adaptive_entropy_score_mode = SCORE_MODE
    probe_args.adaptive_max_raw_start_scale_index = 10
    probe_args.direct_skip_start_scale_index = 100

    scale_schedule = build_scale_schedule(probe_args)
    cfg_list = parse_cfg_list(probe_args.cfg, len(scale_schedule))
    tau_list = parse_tau_list(probe_args.tau, len(scale_schedule))

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

    prompt_reports = []
    for item in prompt_items:
        print(f"[adaptive-score-search] probe prompt={item['name']}", flush=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            elapsed_ms, _ = run_scale_loop_only(
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
            )
        adaptive_report = model.attnController.get_adaptive_segr_report()
        prompt_reports.append(
            {
                "prompt_file": item["name"],
                "prompt_text": item["text"],
                "text_len": item["text_len"],
                "probe_pure_infer_ms": float(elapsed_ms),
                "trace": adaptive_report["trace"],
            }
        )
        model.clear_cache()

    score_band_stats = summarize_score_bands(prompt_reports, lambda_values=lambda_grid)
    search_reports = build_search_reports(
        prompt_reports=prompt_reports,
        lambda_values=lambda_grid,
        extra_gammas=extra_gamma_grid,
        trigger_start_scale_index=args.adaptive_entropy_trigger_start_scale_index,
        max_raw_start=args.adaptive_max_raw_start_scale_index,
        min_effective_start=9,
    )
    preferred_config = select_preferred_config(search_reports)
    selected_prompt_replays = build_selected_prompt_replays(
        prompt_reports=prompt_reports,
        gamma=preferred_config["gamma"],
        lambda_value=preferred_config["lambda"],
        trigger_start_scale_index=args.adaptive_entropy_trigger_start_scale_index,
        max_raw_start=args.adaptive_max_raw_start_scale_index,
        min_effective_start=9,
    )
    preferred_config["prompts"] = selected_prompt_replays

    output = {
        "prompt_files": prompt_files,
        "num_prompts": len(prompt_files),
        "lambda_grid_requested": lambda_grid,
        "gamma_grid_requested": extra_gamma_grid,
        "adaptive_config": {
            "source_scale_index": 8,
            "collect_start_scale_index": int(args.adaptive_entropy_collect_start_scale_index),
            "trigger_start_scale_index": int(args.adaptive_entropy_trigger_start_scale_index),
            "s_min": None if args.adaptive_entropy_s_min is None else int(args.adaptive_entropy_s_min),
            "epsilon": float(args.adaptive_entropy_epsilon),
            "score_mode": SCORE_MODE,
            "score_formula": "abs(delta_h) * candidate_width ** lambda",
            "max_raw_start_scale_index": int(args.adaptive_max_raw_start_scale_index),
            "min_effective_start_scale_index": 9,
            "target_raw_ratios": {str(k): float(v) for k, v in TARGET_RAW_RATIOS.items()},
            "target_effective_ratios": {str(k): float(v) for k, v in TARGET_EFFECTIVE_RATIOS.items()},
        },
        "score_band_stats_by_lambda": score_band_stats,
        "preferred_gamma": preferred_config,
        "preferred_lambda": float(preferred_config["lambda"]),
        "probe_prompts": prompt_reports,
        "lambda_gamma_reports": search_reports,
        "gamma_reports": search_reports,
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[adaptive-score-search] wrote JSON: {report_json}", flush=True)

    if args.report_csv:
        report_csv = Path(args.report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        write_histogram_csv(report_csv, search_reports, len(prompt_files))
        print(f"[adaptive-score-search] wrote CSV: {report_csv}", flush=True)

    del model
    del vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
