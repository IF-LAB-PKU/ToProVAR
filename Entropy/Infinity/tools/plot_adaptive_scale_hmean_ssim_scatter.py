#!/usr/bin/env python3
import argparse
import colorsys
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--min_scale_idx", type=int, default=5)
    parser.add_argument("--max_scale_idx", type=int, default=11)
    return parser.parse_args()


def load_rows(input_csv, min_scale_idx, max_scale_idx):
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("h_mean"):
                continue
            scale_idx = int(row["scale_idx"])
            if scale_idx < int(min_scale_idx) or scale_idx > int(max_scale_idx):
                continue
            rows.append(
                {
                    "prompt_file": row["prompt_file"],
                    "scale_idx": scale_idx,
                    "h_mean": float(row["h_mean"]),
                    "ssim_to_final": float(row["ssim_to_final"]),
                }
            )
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def pearson(xs, ys):
    if len(xs) <= 1 or len(ys) <= 1:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return None
    return num / den


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman(xs, ys):
    if len(xs) <= 1 or len(ys) <= 1:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def linear_fit(xs, ys):
    if len(xs) <= 1:
        return None
    mx = mean(xs)
    my = mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return slope, intercept


def build_summary(rows):
    xs = [row["h_mean"] for row in rows]
    ys = [row["ssim_to_final"] for row in rows]
    by_scale = defaultdict(list)
    by_prompt = defaultdict(list)
    for row in rows:
        by_scale[row["scale_idx"]].append(row)
        by_prompt[row["prompt_file"]].append(row)

    per_scale = []
    for scale_idx in sorted(by_scale):
        group = by_scale[scale_idx]
        gx = [row["h_mean"] for row in group]
        gy = [row["ssim_to_final"] for row in group]
        per_scale.append(
            {
                "scale_idx": scale_idx,
                "num_points": len(group),
                "h_mean_mean": mean(gx),
                "ssim_mean": mean(gy),
                "pearson": pearson(gx, gy) if len(set(gx)) > 1 and len(set(gy)) > 1 else None,
                "spearman": spearman(gx, gy) if len(set(gx)) > 1 and len(set(gy)) > 1 else None,
            }
        )

    return {
        "num_points": len(rows),
        "num_prompts": len(by_prompt),
        "scale_range": [min(row["scale_idx"] for row in rows), max(row["scale_idx"] for row in rows)],
        "overall": {
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "h_mean_range": [min(xs), max(xs)],
            "ssim_range": [min(ys), max(ys)],
        },
        "per_scale": per_scale,
    }


def load_font(size):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def palette_for_scales(scale_values):
    scale_values = sorted(set(scale_values))
    n = max(1, len(scale_values))
    palette = {}
    for idx, scale_idx in enumerate(scale_values):
        hue = idx / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(hue * 0.82, 0.75, 0.9)
        palette[scale_idx] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def map_x(value, x0, x1, data_min, data_max):
    if data_max <= data_min:
        return x0
    ratio = (value - data_min) / (data_max - data_min)
    return x0 + ratio * (x1 - x0)


def map_y(value, y0, y1, data_min, data_max):
    if data_max <= data_min:
        return y1
    ratio = (value - data_min) / (data_max - data_min)
    return y1 - ratio * (y1 - y0)


def draw_centered(draw, box, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) / 2
    y = box[1] + (box[3] - box[1] - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def make_plot(rows, output_png, summary):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    xs = [row["h_mean"] for row in rows]
    ys = [row["ssim_to_final"] for row in rows]
    x_min_raw, x_max_raw = min(xs), max(xs)
    y_min_raw, y_max_raw = min(ys), max(ys)
    x_pad = max(0.15, 0.05 * (x_max_raw - x_min_raw))
    y_pad = max(0.02, 0.05 * (y_max_raw - y_min_raw))
    x_min, x_max = x_min_raw - x_pad, x_max_raw + x_pad
    y_min, y_max = max(0.0, y_min_raw - y_pad), min(1.0, y_max_raw + y_pad)

    width = 1280
    height = 920
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font_title = load_font(28)
    font_header = load_font(17)
    font_text = load_font(14)
    font_small = load_font(12)

    title = (
        f"All-Prompt h_mean vs SSIM Scatter "
        f"(Scale {summary['scale_range'][0]}-{summary['scale_range'][1]})"
    )
    draw_centered(draw, (0, 18, width, 54), title, font_title, (18, 18, 18))

    overall = summary["overall"]
    header = (
        f"n={summary['num_points']} points, prompts={summary['num_prompts']} | "
        f"Pearson={overall['pearson']:.3f}, Spearman={overall['spearman']:.3f} | "
        "Color = scale_idx"
    )
    draw_centered(draw, (0, 56, width, 92), header, font_header, (60, 60, 60))

    plot_left = 92
    plot_top = 132
    plot_right = 1020
    plot_bottom = 796

    draw.rounded_rectangle((plot_left, plot_top, plot_right, plot_bottom), radius=10, outline=(205, 205, 205), fill=(252, 252, 252), width=1)

    for frac in (0.2, 0.4, 0.6, 0.8):
        gx = plot_left + frac * (plot_right - plot_left)
        gy = plot_top + frac * (plot_bottom - plot_top)
        draw.line((gx, plot_top, gx, plot_bottom), fill=(232, 232, 232), width=1)
        draw.line((plot_left, gy, plot_right, gy), fill=(232, 232, 232), width=1)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(120, 120, 120), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(120, 120, 120), width=1)

    palette = palette_for_scales(row["scale_idx"] for row in rows)
    for row in rows:
        px = map_x(row["h_mean"], plot_left, plot_right, x_min, x_max)
        py = map_y(row["ssim_to_final"], plot_top, plot_bottom, y_min, y_max)
        color = palette[row["scale_idx"]]
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(35, 35, 35), width=1)

    fit = linear_fit(xs, ys)
    if fit is not None:
        slope, intercept = fit
        line_x0 = x_min_raw
        line_x1 = x_max_raw
        line_y0 = slope * line_x0 + intercept
        line_y1 = slope * line_x1 + intercept
        px0 = map_x(line_x0, plot_left, plot_right, x_min, x_max)
        py0 = map_y(line_y0, plot_top, plot_bottom, y_min, y_max)
        px1 = map_x(line_x1, plot_left, plot_right, x_min, x_max)
        py1 = map_y(line_y1, plot_top, plot_bottom, y_min, y_max)
        draw.line((px0, py0, px1, py1), fill=(90, 90, 90), width=3)

    draw.text((plot_left - 34, plot_top - 8), f"{y_max:.2f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left - 34, plot_bottom - 8), f"{y_min:.2f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left, plot_bottom + 12), f"{x_min:.2f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_right - 34, plot_bottom + 12), f"{x_max:.2f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left + 8, plot_top - 28), "SSIM", font=font_text, fill=(70, 70, 70))
    draw.text((plot_right - 68, plot_bottom + 28), "h_mean", font=font_text, fill=(70, 70, 70))

    legend_left = 1070
    legend_top = 150
    draw.text((legend_left, legend_top - 28), "Scale Legend", font=font_header, fill=(35, 35, 35))
    legend_y = legend_top
    for scale_idx in sorted(palette):
        color = palette[scale_idx]
        draw.ellipse((legend_left, legend_y, legend_left + 14, legend_y + 14), fill=color, outline=(40, 40, 40), width=1)
        draw.text((legend_left + 22, legend_y - 2), f"scale {scale_idx}", font=font_text, fill=(45, 45, 45))
        legend_y += 28

    detail_y = legend_y + 20
    draw.text((legend_left, detail_y), "Per-scale cross-prompt corr", font=font_header, fill=(35, 35, 35))
    detail_y += 28
    for item in summary["per_scale"]:
        pear = "n/a" if item["pearson"] is None else f"{item['pearson']:.3f}"
        draw.text(
            (legend_left, detail_y),
            f"s{item['scale_idx']}: P={pear}, mean_ssim={item['ssim_mean']:.3f}",
            font=font_small,
            fill=(60, 60, 60),
        )
        detail_y += 20

    footer = "Mixed-prompt scatter is dominated by scale progression; fixed-scale cross-prompt correlation can differ."
    draw_centered(draw, (40, 846, width - 40, 900), footer, font_text, (60, 60, 60))

    image.save(output_png)


def main():
    args = parse_args()
    rows = load_rows(args.input_csv, args.min_scale_idx, args.max_scale_idx)
    if not rows:
        raise ValueError("No rows after h_mean/scale filtering.")

    summary = build_summary(rows)
    make_plot(rows, args.output_png, summary)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved plot to {args.output_png}")


if __name__ == "__main__":
    main()
