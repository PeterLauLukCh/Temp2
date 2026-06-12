"""Benchmark FlashWasserstein semi-dual solvers.

Example:
    PYTHONPATH=FlashWasserstein:code/src python3 FlashWasserstein/benchmarks/bench_semidual.py \
        --sizes 256,512,4096 --dims 8,64 --methods dense,flash,pot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT / "code" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flash_wasserstein import pot_exact_ot, solve_dense_semidual, solve_flash_wasserstein


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("This benchmark requires PyTorch.") from exc
    return torch


def _parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(torch, device):
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _time_call(torch, device, fn, *, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    _sync(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        _sync(torch, device)
        times.append(time.perf_counter() - start)
    peak_mb = _peak_memory_mb(torch, device)
    return result, sum(times) / len(times), min(times), peak_mb


def _result_row(method: str, n: int, d: int, seconds: float, min_seconds: float, peak_mb: float, result):
    return {
        "method": method,
        "n": n,
        "m": n,
        "d": d,
        "seconds_mean": seconds,
        "seconds_min": min_seconds,
        "peak_memory_mb": peak_mb,
        "semidual_value": getattr(result, "semidual_value", None),
        "transport_cost": getattr(result, "transport_cost", None),
        "mass_error_l1": getattr(result, "mass_error_l1", None),
        "n_iter": getattr(result, "n_iter", None),
        "converged": getattr(result, "converged", None),
    }


def run_benchmark(args) -> List[Dict[str, object]]:
    torch = _require_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    methods = {part.strip() for part in args.methods.split(",") if part.strip()}
    sizes = _parse_ints(args.sizes)
    dims = _parse_ints(args.dims)
    rows: List[Dict[str, object]] = []

    for n in sizes:
        for d in dims:
            torch.manual_seed(args.seed)
            x = torch.randn(n, d, device=device)
            y = torch.randn(n, d, device=device)
            print(f"\nProblem n=m={n}, d={d}, device={device}")

            if "dense" in methods and n <= args.dense_max:
                fn = lambda: solve_dense_semidual(
                    x,
                    y,
                    cost_scale=args.cost_scale,
                    max_iter=args.max_iter,
                    lr=args.lr,
                    tol=args.tol,
                    return_history=False,
                )
                result, mean_s, min_s, peak_mb = _time_call(
                    torch, device, fn, warmup=args.warmup, repeat=args.repeat
                )
                row = _result_row("dense", n, d, mean_s, min_s, peak_mb, result)
                print(row)
                rows.append(row)

            if "flash" in methods:
                if device.type != "cuda":
                    print("Skipping flash: CUDA device required.")
                else:
                    fn = lambda: solve_flash_wasserstein(
                        x,
                        y,
                        cost_scale=args.cost_scale,
                        max_iter=args.max_iter,
                        lr=args.lr,
                        tol=args.tol,
                        return_history=False,
                        allow_tf32=not args.no_tf32,
                        autotune=not args.no_autotune,
                    )
                    result, mean_s, min_s, peak_mb = _time_call(
                        torch, device, fn, warmup=args.warmup, repeat=args.repeat
                    )
                    row = _result_row("flash", n, d, mean_s, min_s, peak_mb, result)
                    print(row)
                    rows.append(row)

            if "pot" in methods and n <= args.pot_max:
                try:
                    start = time.perf_counter()
                    pot = pot_exact_ot(x, y, cost_scale=args.cost_scale, max_size=args.pot_max)
                    elapsed = time.perf_counter() - start
                    row = {
                        "method": "pot",
                        "n": n,
                        "m": n,
                        "d": d,
                        "seconds_mean": elapsed,
                        "seconds_min": elapsed,
                        "peak_memory_mb": 0.0,
                        "semidual_value": None,
                        "transport_cost": pot.cost,
                        "mass_error_l1": pot.row_error_l1 + pot.col_error_l1,
                        "n_iter": None,
                        "converged": True,
                    }
                    print(row)
                    rows.append(row)
                except ImportError as exc:
                    print(f"Skipping POT: {exc}")

    return rows


def write_outputs(rows: Iterable[Dict[str, object]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    json_path = output_dir / "bench_semidual.json"
    csv_path = output_dir / "bench_semidual.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256,512,2048")
    parser.add_argument("--dims", default="8,64")
    parser.add_argument("--methods", default="dense,flash,pot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-scale", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dense-max", type=int, default=2048)
    parser.add_argument("--pot-max", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    args = parser.parse_args()
    rows = run_benchmark(args)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
