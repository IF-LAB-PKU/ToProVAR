#!/usr/bin/env python3
import argparse
import json
import re


FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^,]+)")


def parse_log(log_path, n_samples):
    entries = {}
    ordered = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if "[Adaptive Pruning] Triggered" not in line:
                continue
            fields = {
                match.group("key"): match.group("value").strip()
                for match in FIELD_RE.finditer(line)
            }
            if "raw_start" not in fields or "effective_start" not in fields:
                continue
            sample_global_idx = len(ordered)
            prompt_idx = sample_global_idx // n_samples
            sample_idx = sample_global_idx % n_samples
            item = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "raw_start_scale_index": int(fields["raw_start"]),
                "effective_start_scale_index": int(fields["effective_start"]),
                "trigger_scale": fields.get("trigger_scale"),
                "delta_h": fields.get("delta_h"),
                "cost_scale_index": fields.get("cost_scale_index"),
                "cost_n": fields.get("cost_n"),
                "cost_value": fields.get("cost_value"),
                "delta_cost_ratio": fields.get("delta_cost_ratio"),
                "adaptive_score_n": fields.get("adaptive_score_n"),
                "adaptive_score": fields.get("adaptive_score"),
                "adaptive_entropy_lambda": fields.get("adaptive_entropy_lambda"),
                "adaptive_entropy_score_mode": fields.get("adaptive_entropy_score_mode"),
                "reason": fields.get("reason"),
            }
            entries[f"{prompt_idx}:{sample_idx}"] = item
            ordered.append(item)
    return entries, ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    entries, ordered = parse_log(args.log, args.n_samples)
    payload = {
        "format": "prompt_idx:sample_idx",
        "n_samples": args.n_samples,
        "num_entries": len(ordered),
        "entries": entries,
    }
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"wrote {len(ordered)} entries to {args.output}")


if __name__ == "__main__":
    main()
