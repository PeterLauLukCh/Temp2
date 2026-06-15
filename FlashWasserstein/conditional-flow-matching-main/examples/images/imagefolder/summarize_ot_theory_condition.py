"""Summarize the local-vs-global OT condition from benchmark JSON files.

The theory note gives a sufficient condition under which global entropic OT has
lower cost than block-local exact OT:

    W_block - W_global > eps * log(n).

The raw benchmark scripts report unnormalized squared feature costs.  Training
usually uses normalized quadratic costs with scale 1 / (2 * feature_dim), so
this script reports the condition in that normalized geometry by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def iter_rows(root: Path):
    for path in sorted(root.glob("**/global_vs_local*_ot.json")):
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in rows:
            yield path, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Benchmark root containing global_vs_local*_ot.json files")
    parser.add_argument("--eps_values", default="0.01,0.02,0.05")
    parser.add_argument(
        "--cost_scale",
        default="auto",
        help="'auto' uses 1/(2*dim), otherwise pass a numeric scale applied to benchmark costs",
    )
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_csv", default="")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    eps_values = parse_csv_floats(args.eps_values)
    summaries = []

    for path, row in iter_rows(root):
        dim = int(row["dim"])
        scale = 1.0 / (2.0 * dim) if args.cost_scale == "auto" else float(args.cost_scale)
        batch_size = int(row["batch_size"])
        n = batch_size
        raw_gain = float(row["block_local_cost"]) - float(row["global_cost"])
        normalized_gain = raw_gain * scale
        base = {
            "source": str(path),
            "run": path.parent.name,
            "batch_size": batch_size,
            "dim": dim,
            "local_batch": int(row.get("local_batch", 0)),
            "cost_scale": scale,
            "block_cost_normalized": float(row["block_local_cost"]) * scale,
            "global_cost_normalized": float(row["global_cost"]) * scale,
            "context_gain_normalized": normalized_gain,
            "improvement_pct": float(row["global_vs_block_improvement_pct"]),
            "cross_block_fraction": float(row["global_cross_block_fraction"]),
            "block_time_s": float(row["block_total_time_s"]),
            "global_time_s": float(row["global_total_time_s"]),
        }
        for eps in eps_values:
            bias_bound = eps * math.log(n)
            item = {
                **base,
                "eps": eps,
                "eps_log_n": bias_bound,
                "sufficient_margin": normalized_gain - bias_bound,
                "condition_holds": normalized_gain > bias_bound,
            }
            summaries.append(item)

    if not summaries:
        raise FileNotFoundError(f"No global_vs_local*_ot.json files found under {root}")

    for item in summaries:
        print(
            f"{item['run']} eps={item['eps']:g} "
            f"gain={item['context_gain_normalized']:.6f} "
            f"eps_log_n={item['eps_log_n']:.6f} "
            f"margin={item['sufficient_margin']:.6f} "
            f"holds={item['condition_holds']} "
            f"cross={item['cross_block_fraction']:.3f}"
        )

    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summaries, indent=2))
        print(f"wrote {out_json}")

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
