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
    parser.add_argument("--min_scale_idx", type=int, default=7)
    parser.add_argument("--x_min", type=float, default=3.5e-2)
    parser.add_argument("--x_max", type=float, default=8.0e-2)
    return parser.parse_args()


def load_df(input_csv, min_scale_idx):
    df = pd.read_csv(input_csv)
    df = df[df["delta_rel"].notna()].copy()
    df["delta_rel"] = df["delta_rel"].astype(float)
    df["ssim_to_final"] = df["ssim_to_final"].astype(float)
    df["scale_idx"] = df["scale_idx"].astype(int)
    df = df[df["scale_idx"] >= int(min_scale_idx)].copy()
    return df


def maybe_corr(df, method):
    if len(df) <= 1:
        return None
    if df["delta_rel"].nunique() <= 1 or df["ssim_to_final"].nunique() <= 1:
        return None
    return float(df["delta_rel"].corr(df["ssim_to_final"], method=method))


def build_summary(df):
    no_const = df[df["scale_idx"] <= 11].copy()
    summary = {
        "num_points_scale_ge_7": int(len(df)),
        "delta_rel_range": [float(df["delta_rel"].min()), float(df["delta_rel"].max())],
        "delta_rel_quantiles": {
            str(k): float(v)
            for k, v in df["delta_rel"].quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).items()
        },
        "ssim_quantiles": {
            str(k): float(v)
            for k, v in df["ssim_to_final"].quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).items()
        },
        "scale_ge_7_all": {
            "pearson": maybe_corr(df, "pearson"),
            "spearman": maybe_corr(df, "spearman"),
        },
        "scale_7_to_11": {
            "num_points": int(len(no_const)),
            "pearson": maybe_corr(no_const, "pearson"),
            "spearman": maybe_corr(no_const, "spearman"),
        },
        "per_scale": [],
    }
    for scale_idx, group in df.groupby("scale_idx"):
        summary["per_scale"].append(
            {
                "scale_idx": int(scale_idx),
                "num_points": int(len(group)),
                "delta_rel_range": [float(group["delta_rel"].min()), float(group["delta_rel"].max())],
                "ssim_range": [float(group["ssim_to_final"].min()), float(group["ssim_to_final"].max())],
                "pearson": maybe_corr(group, "pearson"),
                "spearman": maybe_corr(group, "spearman"),
            }
        )
    return summary


def make_plot(df, output_png, summary, x_min, x_max):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df_main = df.copy()
    df_no_const = df[df["scale_idx"] <= 11].copy()

    cmap = plt.get_cmap("viridis", df_main["scale_idx"].nunique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200, constrained_layout=True)

    panels = [
        (axes[0], df_main, "Scale 7-12"),
        (axes[1], df_no_const, "Scale 7-11"),
    ]

    for ax, panel_df, title in panels:
        sc = ax.scatter(
            panel_df["delta_rel"],
            panel_df["ssim_to_final"],
            c=panel_df["scale_idx"],
            cmap=cmap,
            s=28,
            alpha=0.82,
            edgecolors="none",
        )
        ax.set_xscale("log")
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(0.15, 1.03)
        ax.set_xlabel("SEGR delta_rel")
        ax.set_ylabel("SSIM to final")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle("Late-Scale SEGR vs SSIM Scatter", fontsize=14)
    note = (
        f"Scale>=7 all: Pearson={summary['scale_ge_7_all']['pearson']:.3f}, "
        f"Spearman={summary['scale_ge_7_all']['spearman']:.3f} | "
        f"Scale 7-11: Pearson={summary['scale_7_to_11']['pearson']:.3f}, "
        f"Spearman={summary['scale_7_to_11']['spearman']:.3f}"
    )
    fig.text(
        0.5,
        0.02,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#cccccc"},
    )
    cbar = fig.colorbar(sc, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("scale_idx")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = load_df(args.input_csv, args.min_scale_idx)
    if df.empty:
        raise ValueError("No rows available after late-scale filtering.")
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
