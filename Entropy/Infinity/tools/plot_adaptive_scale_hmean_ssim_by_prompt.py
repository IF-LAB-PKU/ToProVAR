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
    parser.add_argument("--min_scale_idx", type=int, default=7)
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


def maybe_corr(group, method):
    xs = [row["h_mean"] for row in group]
    ys = [row["ssim_to_final"] for row in group]
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None
    if method == "pearson":
        return pearson(xs, ys)
    if method == "spearman":
        return spearman(xs, ys)
    raise ValueError(f"Unsupported correlation method: {method}")


def build_summary(rows):
    by_prompt = defaultdict(list)
    by_scale = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_file"]].append(row)
        by_scale[row["scale_idx"]].append(row)

    per_prompt = []
    for prompt_file in sorted(by_prompt):
        group = sorted(by_prompt[prompt_file], key=lambda item: item["scale_idx"])
        h_vals = [row["h_mean"] for row in group]
        s_vals = [row["ssim_to_final"] for row in group]
        per_prompt.append(
            {
                "prompt_file": prompt_file,
                "num_points": len(group),
                "h_mean_range": [min(h_vals), max(h_vals)],
                "ssim_range": [min(s_vals), max(s_vals)],
                "pearson": maybe_corr(group, "pearson"),
                "spearman": maybe_corr(group, "spearman"),
            }
        )

    per_scale = []
    for scale_idx in sorted(by_scale):
        group = by_scale[scale_idx]
        per_scale.append(
            {
                "scale_idx": scale_idx,
                "num_points": len(group),
                "pearson": maybe_corr(group, "pearson"),
                "spearman": maybe_corr(group, "spearman"),
            }
        )

    pearsons = [item["pearson"] for item in per_prompt if item["pearson"] is not None]
    spearmans = [item["spearman"] for item in per_prompt if item["spearman"] is not None]
    return {
        "num_prompts": len(per_prompt),
        "num_points": len(rows),
        "pearson_mean": mean(pearsons),
        "pearson_median": median(pearsons),
        "spearman_mean": mean(spearmans),
        "spearman_median": median(spearmans),
        "per_prompt": per_prompt,
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


def padded_bounds(rows):
    h_vals = [row["h_mean"] for row in rows]
    s_vals = [row["ssim_to_final"] for row in rows]
    h_min, h_max = min(h_vals), max(h_vals)
    s_min, s_max = min(s_vals), max(s_vals)
    h_pad = max(0.15, 0.05 * (h_max - h_min))
    s_pad = max(0.02, 0.05 * (s_max - s_min))
    return (
        h_min - h_pad,
        h_max + h_pad,
        max(0.0, s_min - s_pad),
        min(1.0, s_max + s_pad),
    )


def fmt_corr(value):
    return "n/a" if value is None else f"{value:.2f}"


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

    prompt_files = sorted({row["prompt_file"] for row in rows})
    scale_values = sorted({row["scale_idx"] for row in rows})
    palette = palette_for_scales(scale_values)
    x_min, x_max, y_min, y_max = padded_bounds(rows)

    cols = 4
    n = len(prompt_files)
    rows_count = math.ceil(n / cols)
    cell_w = 520
    cell_h = 360
    page_margin = 26
    title_h = 70
    legend_h = 46
    footer_h = 56
    width = page_margin * 2 + cols * cell_w
    height = page_margin * 2 + title_h + legend_h + rows_count * cell_h + footer_h

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = load_font(28)
    font_header = load_font(16)
    font_axis = load_font(14)
    font_text = load_font(13)
    font_small = load_font(11)

    min_scale_idx = min(scale_values)
    max_scale_idx = max(scale_values)
    title = f"Per-Prompt h_mean vs SSIM (Scale {min_scale_idx}-{max_scale_idx})"
    draw_centered(draw, (0, page_margin, width, page_margin + 34), title, font_title, (15, 15, 15))

    summary_line = (
        f"Prompt-wise mean correlation: Pearson={summary['pearson_mean']:.3f}, "
        f"Spearman={summary['spearman_mean']:.3f}"
    )
    draw_centered(draw, (0, page_margin + 34, width, page_margin + title_h), summary_line, font_header, (55, 55, 55))

    legend_y = page_margin + title_h + 6
    legend_x = page_margin
    for scale_idx in scale_values:
        color = palette[scale_idx]
        draw.ellipse((legend_x, legend_y + 4, legend_x + 12, legend_y + 16), fill=color, outline=(90, 90, 90))
        draw.text((legend_x + 18, legend_y), f"scale {scale_idx}", font=font_text, fill=(40, 40, 40))
        legend_x += 84
        if legend_x > width - page_margin - 70:
            legend_x = page_margin
            legend_y += 18

    top_offset = page_margin + title_h + legend_h
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_file"]].append(row)

    for idx, prompt_file in enumerate(prompt_files):
        grid_col = idx % cols
        grid_row = idx // cols
        x0 = page_margin + grid_col * cell_w
        y0 = top_offset + grid_row * cell_h
        x1 = x0 + cell_w - 16
        y1 = y0 + cell_h - 16

        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, outline=(210, 210, 210), fill=(252, 252, 252), width=1)

        group = sorted(by_prompt[prompt_file], key=lambda item: item["scale_idx"])
        plot_left = x0 + 54
        plot_top = y0 + 46
        plot_right = x1 - 20
        plot_bottom = y1 - 42

        for frac in (0.25, 0.5, 0.75):
            gx = plot_left + frac * (plot_right - plot_left)
            gy = plot_top + frac * (plot_bottom - plot_top)
            draw.line((gx, plot_top, gx, plot_bottom), fill=(232, 232, 232), width=1)
            draw.line((plot_left, gy, plot_right, gy), fill=(232, 232, 232), width=1)

        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(120, 120, 120), width=1)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(120, 120, 120), width=1)

        points = []
        for row in group:
            px = map_x(row["h_mean"], plot_left, plot_right, x_min, x_max)
            py = map_y(row["ssim_to_final"], plot_top, plot_bottom, y_min, y_max)
            points.append((px, py, row))

        if len(points) >= 2:
            draw.line([(px, py) for px, py, _ in points], fill=(110, 110, 110), width=2)

        for px, py, row in points:
            color = palette[row["scale_idx"]]
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(40, 40, 40), width=1)
            draw.text((px + 5, py - 8), str(row["scale_idx"]), font=font_small, fill=(35, 35, 35))

        pear = maybe_corr(group, "pearson")
        spear = maybe_corr(group, "spearman")
        header = f"{prompt_file}\nP={fmt_corr(pear)} S={fmt_corr(spear)}"
        draw.multiline_text((x0 + 12, y0 + 10), header, font=font_text, fill=(20, 20, 20), spacing=2)

        draw.text((plot_left - 24, plot_top - 6), f"{y_max:.2f}", font=font_small, fill=(90, 90, 90))
        draw.text((plot_left - 24, plot_bottom - 6), f"{y_min:.2f}", font=font_small, fill=(90, 90, 90))
        draw.text((plot_left, plot_bottom + 8), f"{x_min:.1f}", font=font_small, fill=(90, 90, 90))
        draw.text((plot_right - 28, plot_bottom + 8), f"{x_max:.1f}", font=font_small, fill=(90, 90, 90))
        draw.text((plot_left + 4, plot_top - 22), "SSIM", font=font_axis, fill=(70, 70, 70))
        draw.text((plot_right - 52, plot_bottom + 22), "h_mean", font=font_axis, fill=(70, 70, 70))

    footer = "Points are connected in scale order and labeled by zero-based scale_idx."
    draw_centered(
        draw,
        (page_margin, height - footer_h, width - page_margin, height - 10),
        footer,
        font_text,
        (55, 55, 55),
    )

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
