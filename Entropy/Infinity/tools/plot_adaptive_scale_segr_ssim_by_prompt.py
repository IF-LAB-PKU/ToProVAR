#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--min_scale_idx", type=int, default=7)
    parser.add_argument("--max_scale_idx", type=int, default=11)
    parser.add_argument("--x_min", type=float, default=3.8e-2)
    parser.add_argument("--x_max", type=float, default=6.8e-2)
    return parser.parse_args()


def load_df(input_csv, min_scale_idx, max_scale_idx):
    df = pd.read_csv(input_csv)
    df = df[df["delta_rel"].notna()].copy()
    df["delta_rel"] = df["delta_rel"].astype(float)
    df["ssim_to_final"] = df["ssim_to_final"].astype(float)
    df["scale_idx"] = df["scale_idx"].astype(int)
    return df[(df["scale_idx"] >= int(min_scale_idx)) & (df["scale_idx"] <= int(max_scale_idx))].copy()


def maybe_corr(group, method):
    if len(group) <= 1:
        return None
    if group["delta_rel"].nunique() <= 1 or group["ssim_to_final"].nunique() <= 1:
        return None
    return float(group["delta_rel"].corr(group["ssim_to_final"], method=method))


def build_summary(df):
    rows = []
    for prompt_file, group in df.groupby("prompt_file"):
        group = group.sort_values("scale_idx")
        rows.append(
            {
                "prompt_file": prompt_file,
                "num_points": int(len(group)),
                "delta_rel_range": [float(group["delta_rel"].min()), float(group["delta_rel"].max())],
                "ssim_range": [float(group["ssim_to_final"].min()), float(group["ssim_to_final"].max())],
                "pearson": maybe_corr(group, "pearson"),
                "spearman": maybe_corr(group, "spearman"),
            }
        )

    per_prompt = sorted(rows, key=lambda item: item["prompt_file"])
    pearsons = [item["pearson"] for item in per_prompt if item["pearson"] is not None]
    spearmans = [item["spearman"] for item in per_prompt if item["spearman"] is not None]
    return {
        "num_prompts": len(per_prompt),
        "num_points": int(len(df)),
        "pearson_mean": None if not pearsons else float(pd.Series(pearsons).mean()),
        "pearson_median": None if not pearsons else float(pd.Series(pearsons).median()),
        "spearman_mean": None if not spearmans else float(pd.Series(spearmans).mean()),
        "spearman_median": None if not spearmans else float(pd.Series(spearmans).median()),
        "per_prompt": per_prompt,
    }


def make_plot(df, output_png, summary, x_min, x_max):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(df["prompt_file"].unique().tolist())
    n = len(prompt_files)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), dpi=200, constrained_layout=True)
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis", df["scale_idx"].nunique())

    for ax, prompt_file in zip(axes, prompt_files):
        group = df[df["prompt_file"] == prompt_file].sort_values("scale_idx")
        sc = ax.scatter(
            group["delta_rel"],
            group["ssim_to_final"],
            c=group["scale_idx"],
            cmap=cmap,
            s=38,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
        )
        ax.plot(
            group["delta_rel"],
            group["ssim_to_final"],
            color="#666666",
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
        )
        for _, row in group.iterrows():
            ax.text(
                row["delta_rel"],
                row["ssim_to_final"],
                str(int(row["scale_idx"])),
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.8,
            )
        pearson = maybe_corr(group, "pearson")
        spearman = maybe_corr(group, "spearman")
        ax.set_title(
            f"{prompt_file}\nP={pearson:.2f} S={spearman:.2f}",
            fontsize=9,
        )
        ax.set_xscale("log")
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(0.18, 0.98)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("delta_rel")
        ax.set_ylabel("SSIM")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Per-Prompt Late-Scale SEGR vs SSIM (Scale 7-11)", fontsize=16)
    fig.text(
        0.5,
        0.01,
        (
            f"Prompt-wise mean correlation: Pearson={summary['pearson_mean']:.3f}, "
            f"Spearman={summary['spearman_mean']:.3f} | "
            "Points are connected in scale order and labeled by scale_idx."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#cccccc"},
    )
    cbar = fig.colorbar(sc, ax=axes[:n], shrink=0.9, pad=0.01)
    cbar.set_label("scale_idx")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = load_df(args.input_csv, args.min_scale_idx, args.max_scale_idx)
    if df.empty:
        raise ValueError("No rows after prompt/scale filtering.")
    summary = build_summary(df)
    make_plot(df, args.output_png, summary, x_min=args.x_min, x_max=args.x_max)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved plot to {args.output_png}")


if __name__ == "__main__":
    main()
