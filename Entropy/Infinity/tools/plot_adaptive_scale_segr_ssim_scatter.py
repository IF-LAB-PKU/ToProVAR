#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="")
    return parser.parse_args()


def load_df(input_csv):
    df = pd.read_csv(input_csv)
    df = df[df["delta_rel"].notna()].copy()
    df["delta_rel"] = df["delta_rel"].astype(float)
    df["ssim_to_final"] = df["ssim_to_final"].astype(float)
    df["scale_idx"] = df["scale_idx"].astype(int)
    return df


def correlation_summary(df):
    summary = {
        "num_points": int(len(df)),
        "delta_rel_quantiles": {
            str(k): float(v)
            for k, v in df["delta_rel"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).items()
        },
        "ssim_quantiles": {
            str(k): float(v)
            for k, v in df["ssim_to_final"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).items()
        },
        "overall": {
            "pearson": float(df["delta_rel"].corr(df["ssim_to_final"], method="pearson")),
            "spearman": float(df["delta_rel"].corr(df["ssim_to_final"], method="spearman")),
        },
    }

    no_scale1 = df[df["scale_idx"] >= 2].copy()
    if len(no_scale1) > 1:
        summary["scale_ge_2"] = {
            "num_points": int(len(no_scale1)),
            "pearson": float(no_scale1["delta_rel"].corr(no_scale1["ssim_to_final"], method="pearson")),
            "spearman": float(no_scale1["delta_rel"].corr(no_scale1["ssim_to_final"], method="spearman")),
        }

    per_scale = []
    for scale_idx, group in df.groupby("scale_idx"):
        item = {
            "scale_idx": int(scale_idx),
            "num_points": int(len(group)),
            "delta_rel_mean": float(group["delta_rel"].mean()),
            "ssim_mean": float(group["ssim_to_final"].mean()),
            "pearson": None,
            "spearman": None,
        }
        if len(group) > 1 and group["ssim_to_final"].nunique() > 1 and group["delta_rel"].nunique() > 1:
            item["pearson"] = float(group["delta_rel"].corr(group["ssim_to_final"], method="pearson"))
            item["spearman"] = float(group["delta_rel"].corr(group["ssim_to_final"], method="spearman"))
        per_scale.append(item)
    summary["per_scale"] = per_scale
    return summary


def make_plot(df, output_png, summary):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis", df["scale_idx"].nunique())
    color_values = df["scale_idx"]
    no_scale1 = df[df["scale_idx"] >= 2].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180, constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        df["delta_rel"],
        df["ssim_to_final"],
        c=color_values,
        cmap=cmap,
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("SEGR delta_rel")
    ax.set_ylabel("SSIM to final")
    ax.set_title("All Valid Points (symlog x)")
    ax.grid(alpha=0.25, linewidth=0.6)

    ax = axes[1]
    sc2 = ax.scatter(
        no_scale1["delta_rel"],
        no_scale1["ssim_to_final"],
        c=no_scale1["scale_idx"],
        cmap=cmap,
        s=22,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("SEGR delta_rel")
    ax.set_ylabel("SSIM to final")
    ax.set_title("Scale >= 2 (linear x)")
    ax.grid(alpha=0.25, linewidth=0.6)

    overall = summary["overall"]
    scale_ge_2 = summary.get("scale_ge_2", {})
    text_lines = [
        f"All points: Pearson={overall['pearson']:.3f}, Spearman={overall['spearman']:.3f}",
    ]
    if scale_ge_2:
        text_lines.append(
            f"Scale>=2: Pearson={scale_ge_2['pearson']:.3f}, Spearman={scale_ge_2['spearman']:.3f}"
        )
    text_lines.append("Color = scale_idx")
    fig.text(
        0.5,
        0.02,
        " | ".join(text_lines),
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#cccccc"},
    )

    cbar = fig.colorbar(sc2, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("scale_idx")

    fig.suptitle("Adaptive Scale SEGR vs SSIM Scatter", fontsize=14)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = load_df(args.input_csv)
    summary = correlation_summary(df)
    make_plot(df, args.output_png, summary)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved plot to {args.output_png}")


if __name__ == "__main__":
    main()
