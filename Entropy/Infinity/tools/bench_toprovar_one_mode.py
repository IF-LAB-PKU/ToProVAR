#!/usr/bin/env python3
import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from tools.bench_toprovar_speed import (
    build_scale_schedule,
    load_prompt_items,
    run_benchmark_mode,
)
from tools.run_infinity import (
    add_common_arguments,
    load_tokenizer,
    parse_protected_global_layers,
)


MODE_CONFIGS = {
    "baseline_full": {
        "start_scale_index": 100,
        "direct_skip_start_scale_index": 100,
        "only_scale_skip": False,
        "use_adaptive_scale": False,
        "use_svd_depart_layers": False,
        "use_token_pruning": False,
        "scale_prop": 1.0,
        "structure_layer_ration": 1.0,
        "entropy_boundary_line": 1.0,
        "svd_source_scale_index": 7,
        "svd_split_mode": "ratio",
        "svd_rho_threshold": -1.0,
        "protected_global_layers": [],
    },
    "fixed_src8_start9_r060": {
        "start_scale_index": 9,
        "direct_skip_start_scale_index": 11,
        "only_scale_skip": False,
        "use_adaptive_scale": False,
        "use_svd_depart_layers": True,
        "use_token_pruning": True,
        "scale_prop": 1.0,
        "structure_layer_ration": 0.6,
        "entropy_boundary_line": 1.0,
        "svd_source_scale_index": 8,
        "svd_split_mode": "ratio",
        "svd_rho_threshold": -1.0,
        "protected_global_layers": [],
    },
    "adaptive_src8_score": {
        "start_scale_index": 9,
        "direct_skip_start_scale_index": 11,
        "only_scale_skip": False,
        "use_adaptive_scale": True,
        "use_svd_depart_layers": True,
        "use_token_pruning": True,
        "scale_prop": 1.0,
        "structure_layer_ration": 0.6,
        "entropy_boundary_line": 1.0,
        "svd_source_scale_index": 8,
        "svd_split_mode": "ratio",
        "svd_rho_threshold": -1.0,
        "protected_global_layers": [],
    },
    "adaptive_gamma0055": {},
    "adaptive_gamma0056": {},
}


def natural_key(path_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path_name))
    ]


def load_prompt_file_list(prompt_dir, prompt_files=None, prompt_limit=100):
    if prompt_files:
        files = list(prompt_files)
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
        raise ValueError("No prompt files selected for one-mode benchmark.")
    return files


def apply_mode_config(args):
    mode_args = deepcopy(args)
    bench_mode = args.bench_mode
    if bench_mode in {"adaptive_gamma0055", "adaptive_gamma0056"}:
        bench_mode = "adaptive_src8_score"
    for key, value in MODE_CONFIGS[bench_mode].items():
        setattr(mode_args, key, value)
    if getattr(mode_args, "use_svd_depart_layers", False):
        os.environ.setdefault("TOPROVAR_LAYER_RATIO_SCHEDULE", "9:0.6,10:0.6")
    return mode_args


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--bench_mode", choices=sorted(MODE_CONFIGS), required=True)
    parser.add_argument("--prompt_files", nargs="*", default=None)
    parser.add_argument("--prompt_limit", type=int, default=100)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--timing_stats", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.protected_global_layers = parse_protected_global_layers(args.protected_global_layers)
    return args


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    os.environ["TOPROVAR_ENABLE_TIMING_STATS"] = "1" if args.timing_stats else "0"
    mode_args = apply_mode_config(args)
    prompt_files = load_prompt_file_list(
        prompt_dir=mode_args.prompt_dir,
        prompt_files=args.prompt_files,
        prompt_limit=args.prompt_limit,
    )
    scale_schedule = build_scale_schedule(mode_args)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=mode_args.text_encoder_ckpt)
    prompt_items = load_prompt_items(
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        prompt_dir=mode_args.prompt_dir,
        prompt_files=prompt_files,
    )

    result = run_benchmark_mode(args.bench_mode, mode_args, prompt_items, scale_schedule)
    output = {
        "mode": args.bench_mode,
        "prompt_files": prompt_files,
        "scale_schedule_tokens": [int(np.prod(item)) for item in scale_schedule],
        "pruning_config": {
            "start_scale_index": int(mode_args.start_scale_index),
            "direct_skip_start_scale_index": int(mode_args.direct_skip_start_scale_index),
            "only_scale_skip": bool(mode_args.only_scale_skip),
            "use_adaptive_scale": bool(mode_args.use_adaptive_scale),
            "use_svd_depart_layers": bool(mode_args.use_svd_depart_layers),
            "structure_layer_ration": float(mode_args.structure_layer_ration),
            "use_token_pruning": bool(mode_args.use_token_pruning),
            "svd_source_scale_index": int(mode_args.svd_source_scale_index),
            "svd_split_mode": mode_args.svd_split_mode,
            "svd_rho_threshold": None if mode_args.svd_rho_threshold <= 0 else float(mode_args.svd_rho_threshold),
            "protected_global_layers": parse_protected_global_layers(mode_args.protected_global_layers),
            "adaptive_entropy_collect_start_scale_index": int(mode_args.adaptive_entropy_collect_start_scale_index),
            "adaptive_entropy_trigger_start_scale_index": int(mode_args.adaptive_entropy_trigger_start_scale_index),
            "adaptive_entropy_s_min": None if mode_args.adaptive_entropy_s_min is None else int(mode_args.adaptive_entropy_s_min),
            "adaptive_entropy_gamma": float(mode_args.adaptive_entropy_gamma),
            "adaptive_entropy_epsilon": float(mode_args.adaptive_entropy_epsilon),
            "adaptive_entropy_lambda": float(mode_args.adaptive_entropy_lambda),
            "adaptive_entropy_score_mode": mode_args.adaptive_entropy_score_mode,
            "adaptive_entropy_signal_mode": mode_args.adaptive_entropy_signal_mode,
            "adaptive_entropy_signal_layer_idx": int(mode_args.adaptive_entropy_signal_layer_idx),
            "adaptive_max_raw_start_scale_index": int(mode_args.adaptive_max_raw_start_scale_index),
        },
        "result": result,
        "notes": {
            "pure_infer_boundary": "measured from immediately before the Scale 1 autoregressive loop to immediately after final image tensor materialization",
            "flops_scope": "analytical transformer block FLOPs using prompt-specific kept-token counts",
        },
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"[bench-one-mode] report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
