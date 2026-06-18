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
    parser.add_argument("--min_scale_idx", type=int, default=5)
    parser.add_argument("--max_scale_idx", type=int, default=11)
    parser.add_argument("--ssim_scale_idx", type=int, default=8)
    parser.add_argument(
        "--y_metric",
        type=str,
        default="ssim_scale",
        choices=["ssim_scale", "mean_ssim"],
    )
    parser.add_argument("--y_min_scale_idx", type=int, default=8)
    parser.add_argument("--y_max_scale_idx", type=int, default=10)
    parser.add_argument(
        "--x_metric",
        type=str,
        default="mean_delta_rel",
        choices=["mean_delta_rel", "mean_deltah", "sigmoid_mean_deltah"],
    )
    return parser.parse_args()


def mean(xs):
    return sum(xs) / len(xs) if xs else None


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


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-float(value)))


def load_prompt_scale_rows(input_csv):
    by_prompt = defaultdict(dict)
    with open(input_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt_file = row["prompt_file"]
            scale_idx = int(row["scale_idx"])
            by_prompt[prompt_file][scale_idx] = {
                "h_mean": None if row["h_mean"] == "" else float(row["h_mean"]),
                "delta_rel": None if row["delta_rel"] == "" else float(row["delta_rel"]),
                "ssim_to_final": float(row["ssim_to_final"]),
            }
    return by_prompt


def resolve_x_metric(
    scale_map,
    min_scale_idx,
    max_scale_idx,
    x_metric,
):
    target_scales = list(range(int(min_scale_idx), int(max_scale_idx) + 1))
    if x_metric == "mean_delta_rel":
        missing_scales = []
        delta_values = []
        for scale_idx in target_scales:
            record = scale_map.get(scale_idx)
            if record is None or record["delta_rel"] is None:
                missing_scales.append(scale_idx)
            else:
                delta_values.append(float(record["delta_rel"]))
        if missing_scales:
            return None, f"missing delta_rel for scales {missing_scales}"
        return mean(delta_values), None

    if x_metric in {"mean_deltah", "sigmoid_mean_deltah"}:
        missing_scales = []
        delta_h_values = []
        for scale_idx in target_scales:
            prev_record = scale_map.get(scale_idx - 1)
            record = scale_map.get(scale_idx)
            prev_h = None if prev_record is None else prev_record["h_mean"]
            curr_h = None if record is None else record["h_mean"]
            if prev_h is None or curr_h is None:
                missing_scales.append(scale_idx)
            else:
                delta_h_values.append(abs(float(curr_h) - float(prev_h)))
        if missing_scales:
            return None, f"missing h_mean pair for delta_h at scales {missing_scales}"
        mean_deltah = mean(delta_h_values)
        if x_metric == "mean_deltah":
            return mean_deltah, None
        return sigmoid(mean_deltah), None

    raise ValueError(f"Unsupported x_metric: {x_metric}")


def x_metric_definition(x_metric, min_scale_idx, max_scale_idx):
    if x_metric == "mean_delta_rel":
        return f"mean delta_rel over scale_idx {min_scale_idx}-{max_scale_idx}"
    if x_metric == "mean_deltah":
        return f"mean delta_h over scale_idx {min_scale_idx}-{max_scale_idx}"
    if x_metric == "sigmoid_mean_deltah":
        return f"sigmoid(mean delta_h over scale_idx {min_scale_idx}-{max_scale_idx})"
    raise ValueError(f"Unsupported x_metric: {x_metric}")


def x_metric_column_name(x_metric, min_scale_idx, max_scale_idx):
    if x_metric == "mean_delta_rel":
        return f"avg_delta_rel_{min_scale_idx}_{max_scale_idx}"
    if x_metric == "mean_deltah":
        return f"mean_deltah_{min_scale_idx}_{max_scale_idx}"
    if x_metric == "sigmoid_mean_deltah":
        return f"sigmoid_mean_deltah_{min_scale_idx}_{max_scale_idx}"
    raise ValueError(f"Unsupported x_metric: {x_metric}")


def resolve_y_metric(scale_map, y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx):
    if y_metric == "ssim_scale":
        ssim_record = scale_map.get(int(ssim_scale_idx))
        if ssim_record is None:
            return None, f"missing scale {ssim_scale_idx} row for ssim_to_final"
        return float(ssim_record["ssim_to_final"]), None

    if y_metric == "mean_ssim":
        target_scales = list(range(int(y_min_scale_idx), int(y_max_scale_idx) + 1))
        missing_scales = []
        ssim_values = []
        for scale_idx in target_scales:
            record = scale_map.get(scale_idx)
            if record is None:
                missing_scales.append(scale_idx)
            else:
                ssim_values.append(float(record["ssim_to_final"]))
        if missing_scales:
            return None, f"missing ssim_to_final for scales {missing_scales}"
        return mean(ssim_values), None

    raise ValueError(f"Unsupported y_metric: {y_metric}")


def y_metric_definition(y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx):
    if y_metric == "ssim_scale":
        return f"ssim_to_final at scale_idx {ssim_scale_idx}"
    if y_metric == "mean_ssim":
        return f"mean ssim_to_final over scale_idx {y_min_scale_idx}-{y_max_scale_idx}"
    raise ValueError(f"Unsupported y_metric: {y_metric}")


def y_metric_column_name(y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx):
    if y_metric == "ssim_scale":
        return f"ssim_scale_{ssim_scale_idx}"
    if y_metric == "mean_ssim":
        return f"mean_ssim_{y_min_scale_idx}_{y_max_scale_idx}"
    raise ValueError(f"Unsupported y_metric: {y_metric}")


def aggregate_points(
    by_prompt,
    min_scale_idx,
    max_scale_idx,
    ssim_scale_idx,
    x_metric,
    y_metric,
    y_min_scale_idx,
    y_max_scale_idx,
):
    target_scales = list(range(int(min_scale_idx), int(max_scale_idx) + 1))
    missing_messages = []
    points = []

    for prompt_file in sorted(by_prompt):
        scale_map = by_prompt[prompt_file]
        x_value, x_error = resolve_x_metric(
            scale_map=scale_map,
            min_scale_idx=min_scale_idx,
            max_scale_idx=max_scale_idx,
            x_metric=x_metric,
        )
        y_value, y_error = resolve_y_metric(
            scale_map=scale_map,
            y_metric=y_metric,
            ssim_scale_idx=ssim_scale_idx,
            y_min_scale_idx=y_min_scale_idx,
            y_max_scale_idx=y_max_scale_idx,
        )

        if x_error is not None:
            missing_messages.append(f"{prompt_file}: {x_error}")
            continue
        if y_error is not None:
            missing_messages.append(f"{prompt_file}: {y_error}")
            continue

        points.append(
            {
                "prompt_file": prompt_file,
                "prompt_stem": Path(prompt_file).stem,
                "x_value": float(x_value),
                "y_value": float(y_value),
            }
        )

    if missing_messages:
        raise ValueError("Aggregation failed:\n" + "\n".join(missing_messages))

    return points


def build_summary(
    points,
    min_scale_idx,
    max_scale_idx,
    ssim_scale_idx,
    x_metric,
    y_metric,
    y_min_scale_idx,
    y_max_scale_idx,
):
    xs = [point["x_value"] for point in points]
    ys = [point["y_value"] for point in points]
    fit = linear_fit(xs, ys)
    x_key = x_metric_column_name(x_metric, min_scale_idx, max_scale_idx)
    y_key = y_metric_column_name(y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx)
    summary = {
        "num_points": len(points),
        "num_prompts": len(points),
        "x_metric": x_metric,
        "y_metric": y_metric,
        "x_definition": x_metric_definition(x_metric, min_scale_idx, max_scale_idx),
        "y_definition": y_metric_definition(y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx),
        "overall": {
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
        },
        "x_range": [min(xs), max(xs)],
        "y_range": [min(ys), max(ys)],
        "per_prompt": [
            {
                "prompt_file": point["prompt_file"],
                "prompt_stem": point["prompt_stem"],
                x_key: point["x_value"],
                y_key: point["y_value"],
            }
            for point in points
        ],
    }
    if fit is not None:
        summary["linear_fit"] = {
            "slope": fit[0],
            "intercept": fit[1],
        }
    return summary


def write_points_csv(
    points,
    output_points_csv,
    min_scale_idx,
    max_scale_idx,
    ssim_scale_idx,
    x_metric,
    y_metric,
    y_min_scale_idx,
    y_max_scale_idx,
):
    output_points_csv = Path(output_points_csv)
    output_points_csv.parent.mkdir(parents=True, exist_ok=True)
    x_key = x_metric_column_name(x_metric, min_scale_idx, max_scale_idx)
    y_key = y_metric_column_name(y_metric, ssim_scale_idx, y_min_scale_idx, y_max_scale_idx)
    with output_points_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_file", "prompt_stem", x_key, y_key],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "prompt_file": point["prompt_file"],
                    "prompt_stem": point["prompt_stem"],
                    x_key: point["x_value"],
                    y_key: point["y_value"],
                }
            )


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


def draw_centered(draw, box, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) / 2
    y = box[1] + (box[3] - box[1] - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


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


def make_plot(
    points,
    output_png,
    summary,
    min_scale_idx,
    max_scale_idx,
    ssim_scale_idx,
    x_metric,
    y_metric,
    y_min_scale_idx,
    y_max_scale_idx,
):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    xs = [point["x_value"] for point in points]
    ys = [point["y_value"] for point in points]
    x_min_raw, x_max_raw = min(xs), max(xs)
    y_min_raw, y_max_raw = min(ys), max(ys)
    x_pad = max(0.0005, 0.08 * (x_max_raw - x_min_raw))
    y_pad = max(0.02, 0.08 * (y_max_raw - y_min_raw))
    x_min, x_max = x_min_raw - x_pad, x_max_raw + x_pad
    y_min, y_max = max(0.0, y_min_raw - y_pad), min(1.0, y_max_raw + y_pad)

    width = 1600
    height = 1040
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font_title = load_font(28)
    font_header = load_font(17)
    font_text = load_font(14)
    font_small = load_font(12)

    title = f"{summary['x_definition']} vs {summary['y_definition']}"
    draw_centered(draw, (0, 18, width, 56), title, font_title, (18, 18, 18))

    overall = summary["overall"]
    header = (
        f"n={summary['num_points']} | Pearson={overall['pearson']:.3f} | "
        f"Spearman={overall['spearman']:.3f} | "
        f"x={x_metric} | y={y_metric}"
    )
    draw_centered(draw, (0, 58, width, 94), header, font_header, (58, 58, 58))

    plot_left = 110
    plot_top = 134
    plot_right = 1235
    plot_bottom = 870

    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        radius=10,
        outline=(205, 205, 205),
        fill=(252, 252, 252),
        width=1,
    )

    for frac in (0.2, 0.4, 0.6, 0.8):
        gx = plot_left + frac * (plot_right - plot_left)
        gy = plot_top + frac * (plot_bottom - plot_top)
        draw.line((gx, plot_top, gx, plot_bottom), fill=(232, 232, 232), width=1)
        draw.line((plot_left, gy, plot_right, gy), fill=(232, 232, 232), width=1)

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
            fill=(80, 80, 80),
            width=3,
        )

    point_color = (33, 113, 181)
    label_color = (30, 30, 30)
    for idx, point in enumerate(points):
        px = map_x(point["x_value"], plot_left, plot_right, x_min, x_max)
        py = map_y(point["y_value"], plot_top, plot_bottom, y_min, y_max)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=point_color, outline=(30, 30, 30), width=1)
        label_dx = 8 if idx % 2 == 0 else -8
        label_anchor = "la" if label_dx > 0 else "ra"
        draw.text(
            (px + label_dx, py - 10),
            point["prompt_stem"],
            font=font_small,
            fill=label_color,
            anchor=label_anchor,
        )

    draw.text((plot_left - 44, plot_top - 8), f"{y_max:.3f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left - 44, plot_bottom - 8), f"{y_min:.3f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left, plot_bottom + 14), f"{x_min:.4f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_right - 52, plot_bottom + 14), f"{x_max:.4f}", font=font_small, fill=(95, 95, 95))
    draw.text((plot_left + 8, plot_top - 28), y_metric, font=font_text, fill=(70, 70, 70))
    draw.text((plot_right - 250, plot_bottom + 32), x_metric, font=font_text, fill=(70, 70, 70))

    text_left = 1290
    text_top = 154
    draw.text((text_left, text_top), "Definitions", font=font_header, fill=(28, 28, 28))
    draw.text(
        (text_left, text_top + 32),
        f"x: {summary['x_definition']}",
        font=font_small,
        fill=(55, 55, 55),
    )
    draw.text(
        (text_left, text_top + 56),
        f"y: {summary['y_definition']}",
        font=font_small,
        fill=(55, 55, 55),
    )

    draw.text((text_left, text_top + 110), "Ranges", font=font_header, fill=(28, 28, 28))
    draw.text(
        (text_left, text_top + 142),
        f"x range: [{summary['x_range'][0]:.6f}, {summary['x_range'][1]:.6f}]",
        font=font_small,
        fill=(55, 55, 55),
    )
    draw.text(
        (text_left, text_top + 166),
        f"y range: [{summary['y_range'][0]:.6f}, {summary['y_range'][1]:.6f}]",
        font=font_small,
        fill=(55, 55, 55),
    )

    draw.text((text_left, text_top + 220), "Linear Fit", font=font_header, fill=(28, 28, 28))
    if fit is None:
        draw.text((text_left, text_top + 252), "unavailable", font=font_small, fill=(55, 55, 55))
    else:
        draw.text(
            (text_left, text_top + 252),
            f"slope: {fit[0]:.6f}",
            font=font_small,
            fill=(55, 55, 55),
        )
        draw.text(
            (text_left, text_top + 276),
            f"intercept: {fit[1]:.6f}",
            font=font_small,
            fill=(55, 55, 55),
        )

    footer = "Each point is one prompt aggregated from the formal 20-prompt metrics_per_scale.csv."
    draw_centered(draw, (40, 930, width - 40, 992), footer, font_text, (60, 60, 60))

    image.save(output_png)


def main():
    args = parse_args()
    by_prompt = load_prompt_scale_rows(args.input_csv)
    points = aggregate_points(
        by_prompt=by_prompt,
        min_scale_idx=args.min_scale_idx,
        max_scale_idx=args.max_scale_idx,
        ssim_scale_idx=args.ssim_scale_idx,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        y_min_scale_idx=args.y_min_scale_idx,
        y_max_scale_idx=args.y_max_scale_idx,
    )
    summary = build_summary(
        points=points,
        min_scale_idx=args.min_scale_idx,
        max_scale_idx=args.max_scale_idx,
        ssim_scale_idx=args.ssim_scale_idx,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        y_min_scale_idx=args.y_min_scale_idx,
        y_max_scale_idx=args.y_max_scale_idx,
    )

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_points_csv(
        points=points,
        output_points_csv=args.output_points_csv,
        min_scale_idx=args.min_scale_idx,
        max_scale_idx=args.max_scale_idx,
        ssim_scale_idx=args.ssim_scale_idx,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        y_min_scale_idx=args.y_min_scale_idx,
        y_max_scale_idx=args.y_max_scale_idx,
    )
    make_plot(
        points=points,
        output_png=output_png,
        summary=summary,
        min_scale_idx=args.min_scale_idx,
        max_scale_idx=args.max_scale_idx,
        ssim_scale_idx=args.ssim_scale_idx,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        y_min_scale_idx=args.y_min_scale_idx,
        y_max_scale_idx=args.y_max_scale_idx,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved plot to {output_png}")
    print(f"saved summary to {summary_path}")
    print(f"saved points to {args.output_points_csv}")


if __name__ == "__main__":
    main()
