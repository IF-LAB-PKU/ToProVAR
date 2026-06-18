#!/usr/bin/env python3
import argparse
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
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_points_csv", type=str, required=True)
    parser.add_argument("--scales", type=str, default="8,9,10")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    if len(xs) <= 1 or len(ys) <= 1:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
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
    x_mean = mean(xs)
    y_mean = mean(ys)
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def zscore(values):
    value_mean = mean(values)
    variance = mean([(value - value_mean) ** 2 for value in values])
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - value_mean) / std for value in values]


def load_scale_rows(input_csv):
    by_prompt = defaultdict(dict)
    with open(input_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scale_idx = int(row["scale_idx"])
            by_prompt[row["prompt_file"]][scale_idx] = {
                "h_mean": None if row["h_mean"] == "" else float(row["h_mean"]),
                "delta_rel": None if row["delta_rel"] == "" else float(row["delta_rel"]),
                "ssim_to_final": float(row["ssim_to_final"]),
            }
    return by_prompt


def build_points(by_prompt, scales):
    points = []
    missing = []
    for prompt_file in sorted(by_prompt):
        scale_map = by_prompt[prompt_file]
        for scale_idx in scales:
            prev_scale_idx = scale_idx - 1
            curr = scale_map.get(scale_idx)
            prev = scale_map.get(prev_scale_idx)
            if curr is None or prev is None:
                missing.append(f"{prompt_file}: missing scale {prev_scale_idx} or {scale_idx}")
                continue
            curr_h = curr["h_mean"]
            prev_h = prev["h_mean"]
            curr_ssim = curr["ssim_to_final"]
            prev_ssim = prev["ssim_to_final"]
            curr_delta_rel = curr["delta_rel"]
            if curr_h is None or prev_h is None or curr_delta_rel is None:
                missing.append(f"{prompt_file}: missing h_mean/delta_rel at scale {scale_idx}")
                continue
            points.append(
                {
                    "prompt_file": prompt_file,
                    "prompt_stem": Path(prompt_file).stem,
                    "scale_idx": int(scale_idx),
                    "delta_h": abs(float(curr_h) - float(prev_h)),
                    "delta_rel": float(curr_delta_rel),
                    "ssim": float(curr_ssim),
                    "delta_ssim": float(curr_ssim) - float(prev_ssim),
                    "remain": 1.0 - float(curr_ssim),
                }
            )
    if missing:
        raise ValueError("Failed to build local points:\n" + "\n".join(missing))
    return points


def add_standardized_fields(points, scales):
    grouped = defaultdict(list)
    for point in points:
        grouped[point["scale_idx"]].append(point)

    for scale_idx in scales:
        bucket = grouped[scale_idx]
        delta_h_z = zscore([point["delta_h"] for point in bucket])
        delta_ssim_z = zscore([point["delta_ssim"] for point in bucket])
        for point, xz, yz in zip(bucket, delta_h_z, delta_ssim_z):
            point["z_delta_h"] = xz
            point["z_delta_ssim"] = yz


def summarize(points, scales):
    per_scale = []
    for scale_idx in scales:
        bucket = [point for point in points if point["scale_idx"] == scale_idx]
        xs = [point["delta_h"] for point in bucket]
        ys = [point["delta_ssim"] for point in bucket]
        per_scale.append(
            {
                "scale_idx": int(scale_idx),
                "num_points": len(bucket),
                "delta_h_range": [min(xs), max(xs)],
                "delta_ssim_range": [min(ys), max(ys)],
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    z_x = [point["z_delta_h"] for point in points]
    z_y = [point["z_delta_ssim"] for point in points]
    return {
        "num_points": len(points),
        "num_prompts": len({point["prompt_file"] for point in points}),
        "scales": [int(scale_idx) for scale_idx in scales],
        "definitions": {
            "x_raw": "delta_h = |h_mean[s] - h_mean[s-1]|",
            "y_raw": "delta_ssim = ssim_to_final[s] - ssim_to_final[s-1]",
            "x_pooled": "zscore(delta_h) computed within each scale, then pooled",
            "y_pooled": "zscore(delta_ssim) computed within each scale, then pooled",
        },
        "pooled_standardized": {
            "pearson": pearson(z_x, z_y),
            "spearman": spearman(z_x, z_y),
            "x_range": [min(z_x), max(z_x)],
            "y_range": [min(z_y), max(z_y)],
        },
        "per_scale_raw": per_scale,
    }


def write_points_csv(points, output_points_csv):
    output_points_csv = Path(output_points_csv)
    output_points_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_points_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_file",
                "prompt_stem",
                "scale_idx",
                "delta_h",
                "delta_rel",
                "ssim",
                "delta_ssim",
                "remain",
                "z_delta_h",
                "z_delta_ssim",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(point)


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


def draw_scatter_panel(draw, points, box, x_key, y_key, title, label, point_color, fonts):
    x0, y0, x1, y1 = box
    plot_left = x0 + 56
    plot_top = y0 + 46
    plot_right = x1 - 18
    plot_bottom = y1 - 44
    font_header, font_text, font_small = fonts

    xs = [point[x_key] for point in points]
    ys = [point[y_key] for point in points]
    x_min_raw, x_max_raw = min(xs), max(xs)
    y_min_raw, y_max_raw = min(ys), max(ys)
    x_pad = max(0.02 * max(1.0, abs(x_max_raw - x_min_raw)), 0.06 * (x_max_raw - x_min_raw))
    y_pad = max(0.02 * max(1.0, abs(y_max_raw - y_min_raw)), 0.08 * (y_max_raw - y_min_raw))
    x_min = x_min_raw - x_pad
    x_max = x_max_raw + x_pad
    y_min = y_min_raw - y_pad
    y_max = y_max_raw + y_pad

    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, outline=(210, 210, 210), fill=(252, 252, 252), width=1)
    draw.text((x0 + 14, y0 + 10), title, font=font_header, fill=(20, 20, 20))

    for frac in (0.25, 0.5, 0.75):
        gx = plot_left + frac * (plot_right - plot_left)
        gy = plot_top + frac * (plot_bottom - plot_top)
        draw.line((gx, plot_top, gx, plot_bottom), fill=(233, 233, 233), width=1)
        draw.line((plot_left, gy, plot_right, gy), fill=(233, 233, 233), width=1)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(120, 120, 120), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(120, 120, 120), width=1)

    fit = linear_fit(xs, ys)
    if fit is not None:
        slope, intercept = fit
        line_x0 = x_min_raw
        line_x1 = x_max_raw
        line_y0 = slope * line_x0 + intercept
        line_y1 = slope * line_x1 + intercept
        draw.line(
            (
                map_x(line_x0, plot_left, plot_right, x_min, x_max),
                map_y(line_y0, plot_top, plot_bottom, y_min, y_max),
                map_x(line_x1, plot_left, plot_right, x_min, x_max),
                map_y(line_y1, plot_top, plot_bottom, y_min, y_max),
            ),
            fill=(82, 82, 82),
            width=3,
        )

    for point in points:
        px = map_x(point[x_key], plot_left, plot_right, x_min, x_max)
        py = map_y(point[y_key], plot_top, plot_bottom, y_min, y_max)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=point_color, outline=(34, 34, 34), width=1)

    p = pearson(xs, ys)
    s = spearman(xs, ys)
    draw.text((x0 + 14, y1 - 26), f"P={p:.3f}  S={s:.3f}  n={len(points)}", font=font_small, fill=(45, 45, 45))
    draw.text((plot_left + 6, plot_top - 24), label["y"], font=font_text, fill=(75, 75, 75))
    draw.text((plot_right - 140, plot_bottom + 20), label["x"], font=font_text, fill=(75, 75, 75))
    draw.text((plot_left - 42, plot_top - 8), f"{y_max:.3f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left - 42, plot_bottom - 8), f"{y_min:.3f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left, plot_bottom + 6), f"{x_min:.3f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_right - 42, plot_bottom + 6), f"{x_max:.3f}", font=font_small, fill=(95, 95, 95))


def make_plot(points, output_png, summary, scales):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    width = 1600
    height = 1120
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font_title = load_font(28)
    font_header = load_font(18)
    font_text = load_font(14)
    font_small = load_font(12)

    title = "Local 8/9/10: entropy growth vs SSIM gain"
    draw_centered(draw, (0, 18, width, 54), title, font_title, (18, 18, 18))

    pooled = summary["pooled_standardized"]
    subtitle = (
        f"Best local view: zscore(delta_h) vs zscore(delta_ssim) within each scale, pooled over {scales} | "
        f"Pearson={pooled['pearson']:.3f}, Spearman={pooled['spearman']:.3f}"
    )
    draw_centered(draw, (0, 58, width, 92), subtitle, font_text, (55, 55, 55))

    panels = [
        (60, 130, 780, 560),
        (820, 130, 1540, 560),
        (60, 600, 780, 1030),
        (820, 600, 1540, 1030),
    ]
    colors = {
        scales[0]: (49, 130, 189),
        scales[1]: (49, 163, 84),
        scales[2]: (222, 119, 174),
        "pooled": (99, 99, 99),
    }

    for panel_idx, scale_idx in enumerate(scales):
        bucket = [point for point in points if point["scale_idx"] == scale_idx]
        draw_scatter_panel(
            draw=draw,
            points=bucket,
            box=panels[panel_idx],
            x_key="delta_h",
            y_key="delta_ssim",
            title=f"Scale {scale_idx}: raw delta_h vs raw ΔSSIM",
            label={"x": "delta_h", "y": "delta_ssim"},
            point_color=colors[scale_idx],
            fonts=(font_header, font_text, font_small),
        )

    draw_scatter_panel(
        draw=draw,
        points=points,
        box=panels[3],
        x_key="z_delta_h",
        y_key="z_delta_ssim",
        title="Pooled 8/9/10 after per-scale standardization",
        label={"x": "z_delta_h", "y": "z_delta_ssim"},
        point_color=colors["pooled"],
        fonts=(font_header, font_text, font_small),
    )

    footer = (
        "Raw pooled 8/9/10 mixes different scale distributions and hides the relation. "
        "Per-scale raw panels and pooled z-score panel remove that confound."
    )
    draw_centered(draw, (40, 1052, width - 40, 1096), footer, font_text, (60, 60, 60))
    image.save(output_png)


def main():
    args = parse_args()
    scales = [int(item) for item in args.scales.split(",") if item.strip()]
    if len(scales) != 3:
        raise ValueError(f"Expected exactly 3 scales, got {scales}")

    by_prompt = load_scale_rows(args.input_csv)
    points = build_points(by_prompt, scales)
    add_standardized_fields(points, scales)
    summary = summarize(points, scales)

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_points_csv(points, args.output_points_csv)
    make_plot(points, output_png, summary, scales)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved plot to {output_png}")
    print(f"saved summary to {summary_path}")
    print(f"saved points to {args.output_points_csv}")


if __name__ == "__main__":
    main()
