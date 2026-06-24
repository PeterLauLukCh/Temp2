"""Summarize CIFAR-10 eval JSON files as mean/std over seeds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def method_label(run: str) -> str:
    if run.startswith("local_exact_pot"):
        return "Local exact POT, ctx 128"
    match = re.match(r"flash_global_entropic_ctx(\d+)_eps([0-9.]+)_", run)
    if match:
        ctx = int(match.group(1))
        eps = match.group(2)
        if ctx >= 1024:
            ctx_label = f"{ctx // 1024}K"
        else:
            ctx_label = str(ctx)
        return f"Flash, ctx {ctx_label}, eps {eps}"
    return run


def seed_from_run(run: str) -> int:
    match = re.search(r"_seed(\d+)$", run)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser()
    rows = []
    for path in sorted(eval_root.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    if not rows:
        raise SystemExit(f"No JSON eval rows found in {eval_root}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(method_label(row["run"]), int(row["integration_steps"]))].append(float(row["fid"]))

    summary_rows = []
    for (method, nfe), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append(
            {
                "method": method,
                "nfe": nfe,
                "n": len(values),
                "fid_mean": mean(values),
                "fid_std": stdev(values) if len(values) > 1 else 0.0,
                "fid_values": ";".join(f"{value:.6f}" for value in values),
            }
        )

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else eval_root / "variance_summary.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "nfe", "n", "fid_mean", "fid_std", "fid_values"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {out_csv}")

    if args.markdown:
        nfes = sorted({row["nfe"] for row in summary_rows})
        methods = sorted({row["method"] for row in summary_rows})
        lookup = {(row["method"], row["nfe"]): row for row in summary_rows}
        print()
        print("| Method | " + " | ".join(f"NFE {nfe}" for nfe in nfes) + " |")
        print("|---|" + "|".join("---:" for _ in nfes) + "|")
        for method in methods:
            cells = []
            for nfe in nfes:
                row = lookup.get((method, nfe))
                if row is None:
                    cells.append("")
                else:
                    cells.append(f"{row['fid_mean']:.3f} +/- {row['fid_std']:.3f}")
            print("| " + method + " | " + " | ".join(cells) + " |")

    print()
    for row in summary_rows:
        print(
            f"{row['method']:32s} NFE={row['nfe']:4d} "
            f"n={row['n']} FID={row['fid_mean']:.4f}+/-{row['fid_std']:.4f}"
        )


if __name__ == "__main__":
    main()
