"""2D Gaussian benchmarks for FlashWasserstein.

The benchmark compares:
  - Flash semi-dual solver using the streaming hard c-transform,
  - dense semi-dual solver on small/medium sizes,
  - POT exact Kantorovich OT on small sizes.

The POT cost is a feasible Kantorovich reference. The deterministic Monge
assignment reported by Flash/Dense can have lower raw cost when it violates the
target marginal, so always read it together with ``mass_error_l1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT / "code" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flash_wasserstein import pot_exact_ot, solve_dense_semidual, solve_flash_wasserstein


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("This benchmark requires PyTorch.") from exc
    return torch


def parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def gaussian_specs(case: str):
    if case == "shift":
        return (
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [2.0, -1.0],
            [[1.2, 0.45], [0.45, 0.8]],
        )
    if case == "anisotropic":
        return (
            [0.0, 0.0],
            [[0.2, 0.0], [0.0, 2.0]],
            [0.5, 0.8],
            [[2.0, 0.75], [0.75, 0.5]],
        )
    if case == "near":
        return (
            [0.0, 0.0],
            [[1.0, 0.35], [0.35, 1.0]],
            [0.25, -0.2],
            [[1.1, 0.25], [0.25, 0.9]],
        )
    raise ValueError('case must be one of {"shift", "anisotropic", "near"}.')


def gaussian_population_cost(torch, case: str, *, cost_scale: float) -> float:
    """Closed-form OT cost between the population Gaussians."""

    mean_x, cov_x, mean_y, cov_y = gaussian_specs(case)
    mean_x_t = torch.tensor(mean_x, dtype=torch.float64)
    mean_y_t = torch.tensor(mean_y, dtype=torch.float64)
    cov_x_t = torch.tensor(cov_x, dtype=torch.float64)
    cov_y_t = torch.tensor(cov_y, dtype=torch.float64)

    def sym_sqrt(mat):
        evals, evecs = torch.linalg.eigh((mat + mat.T) * 0.5)
        evals = evals.clamp_min(0.0)
        return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T

    cov_y_sqrt = sym_sqrt(cov_y_t)
    middle = cov_y_sqrt @ cov_x_t @ cov_y_sqrt
    w2_sq = (mean_x_t - mean_y_t).pow(2).sum()
    w2_sq = w2_sq + torch.trace(cov_x_t + cov_y_t - 2.0 * sym_sqrt(middle))
    return float(cost_scale * w2_sq.item())


def gaussian_case(torch, n: int, case: str, *, seed: int, device):
    """Generate a pair of 2D Gaussian clouds."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    mean_x, cov_x, mean_y, cov_y = gaussian_specs(case)

    def sample(mean, cov):
        mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
        cov_t = torch.tensor(cov, device=device, dtype=torch.float32)
        chol = torch.linalg.cholesky(cov_t)
        z = torch.randn(n, 2, device=device, generator=generator)
        return z @ chol.T + mean_t

    return sample(mean_x, cov_x), sample(mean_y, cov_y)


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(torch, device):
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def timed(torch, device, fn):
    sync(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    sync(torch, device)
    elapsed = time.perf_counter() - start
    return result, elapsed, peak_memory_mb(torch, device)


def effective_lr(args, n: int) -> float:
    if args.lr_mass_scale is None:
        return float(args.lr)
    return float(args.lr_mass_scale) * float(n)


def result_row(
    method: str,
    case: str,
    n: int,
    seed: int,
    lr_value: float,
    seconds: float,
    peak_mb: float,
    result,
):
    return {
        "method": method,
        "case": case,
        "n": n,
        "seed": seed,
        "lr": lr_value,
        "seconds": seconds,
        "peak_memory_mb": peak_mb,
        "semidual_value": getattr(result, "semidual_value", None),
        "transport_cost": getattr(result, "transport_cost", None),
        "mass_error_l1": getattr(result, "mass_error_l1", None),
        "n_iter": getattr(result, "n_iter", None),
        "converged": getattr(result, "converged", None),
    }


def run(args):
    torch = require_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    rows: List[Dict[str, object]] = []
    sizes = parse_csv_ints(args.sizes)
    cases = parse_csv_strings(args.cases)
    seeds = parse_csv_ints(args.seeds)

    for case in cases:
        for n in sizes:
            for seed in seeds:
                x, y = gaussian_case(torch, n, case, seed=seed, device=device)
                lr_value = effective_lr(args, n)
                print(f"\ncase={case} n={n} seed={seed} device={device} lr={lr_value:g}")

                dense_result = None
                if n <= args.dense_max:
                    def dense_fn(max_iter=args.max_iter):
                        return solve_dense_semidual(
                            x,
                            y,
                            cost_scale=args.cost_scale,
                            max_iter=max_iter,
                            lr=lr_value,
                            tol=args.tol,
                            return_history=False,
                        )

                    if args.warmup_iter > 0:
                        dense_fn(max_iter=args.warmup_iter)

                    dense_result, elapsed, peak = timed(torch, device, dense_fn)
                    row = result_row("dense", case, n, seed, lr_value, elapsed, peak, dense_result)
                    print(row)
                    rows.append(row)

                flash_result = None
                if device.type == "cuda":
                    def flash_fn(max_iter=args.max_iter):
                        return solve_flash_wasserstein(
                            x,
                            y,
                            cost_scale=args.cost_scale,
                            max_iter=max_iter,
                            lr=lr_value,
                            tol=args.tol,
                            return_history=False,
                            allow_tf32=not args.no_tf32,
                            autotune=not args.no_autotune,
                        )

                    if args.warmup_iter > 0:
                        flash_fn(max_iter=args.warmup_iter)

                    flash_result, elapsed, peak = timed(torch, device, flash_fn)
                    row = result_row("flash", case, n, seed, lr_value, elapsed, peak, flash_result)
                    if dense_result is not None:
                        row["abs_semidual_vs_dense"] = abs(
                            flash_result.semidual_value - dense_result.semidual_value
                        )
                        row["abs_transport_vs_dense"] = abs(
                            flash_result.transport_cost - dense_result.transport_cost
                        )
                        row["abs_mass_error_vs_dense"] = abs(
                            flash_result.mass_error_l1 - dense_result.mass_error_l1
                        )
                    print(row)
                    rows.append(row)
                else:
                    print("Skipping flash: CUDA device required.")

                if n <= args.pot_max:
                    try:
                        fn = lambda: pot_exact_ot(
                            x,
                            y,
                            cost_scale=args.cost_scale,
                            max_size=args.pot_max,
                        )
                        pot_result, elapsed, peak = timed(torch, device, fn)
                        row = {
                            "method": "pot_exact",
                            "case": case,
                            "n": n,
                            "seed": seed,
                            "lr": None,
                            "seconds": elapsed,
                            "peak_memory_mb": peak,
                            "semidual_value": None,
                            "transport_cost": pot_result.cost,
                            "mass_error_l1": pot_result.row_error_l1 + pot_result.col_error_l1,
                            "n_iter": None,
                            "converged": True,
                        }
                        if flash_result is not None:
                            row["flash_transport_minus_pot"] = (
                                flash_result.transport_cost - pot_result.cost
                            )
                            row["flash_mass_error_l1"] = flash_result.mass_error_l1
                        print(row)
                        rows.append(row)
                    except ImportError as exc:
                        print(f"Skipping POT: {exc}")

                if args.gaussian_reference:
                    gaussian_cost = gaussian_population_cost(
                        torch,
                        case,
                        cost_scale=args.cost_scale,
                    )
                    row = {
                        "method": "gaussian_closed_form",
                        "case": case,
                        "n": n,
                        "seed": seed,
                        "lr": None,
                        "seconds": 0.0,
                        "peak_memory_mb": 0.0,
                        "semidual_value": None,
                        "transport_cost": gaussian_cost,
                        "mass_error_l1": 0.0,
                        "n_iter": None,
                        "converged": True,
                    }
                    if flash_result is not None:
                        row["flash_semidual_minus_gaussian"] = (
                            flash_result.semidual_value - gaussian_cost
                        )
                    print(row)
                    rows.append(row)
    return rows


def summarize(rows: List[Dict[str, object]]):
    by_key: Dict[tuple, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (row["case"], row["n"], row["seed"])
        by_key.setdefault(key, {})[row["method"]] = row

    print("\nSummary")
    print("case,n,seed,lr,flash_s,dense_s,pot_s,flash_mass,flash_semidual,flash_cost,pot_cost,gaussian_cost")
    for key, methods in sorted(by_key.items()):
        flash = methods.get("flash")
        dense = methods.get("dense")
        pot = methods.get("pot_exact")
        gauss = methods.get("gaussian_closed_form")
        print(
            f"{key[0]},{key[1]},{key[2]},"
            f"{None if flash is None else flash.get('lr')},"
            f"{None if flash is None else flash.get('seconds')},"
            f"{None if dense is None else dense.get('seconds')},"
            f"{None if pot is None else pot.get('seconds')},"
            f"{None if flash is None else flash.get('mass_error_l1')},"
            f"{None if flash is None else flash.get('semidual_value')},"
            f"{None if flash is None else flash.get('transport_cost')},"
            f"{None if pot is None else pot.get('transport_cost')},"
            f"{None if gauss is None else gauss.get('transport_cost')}"
        )


def write_outputs(rows: List[Dict[str, object]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gaussian_2d_results.json"
    csv_path = output_dir / "gaussian_2d_results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256,512,2048,8192")
    parser.add_argument("--cases", default="shift,anisotropic,near")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-scale", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument(
        "--lr-mass-scale",
        type=float,
        default=None,
        help="If set, use lr = lr_mass_scale * n for each problem size.",
    )
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-iter",
        type=int,
        default=0,
        help="Untimed warmup solver iterations per method/shape.",
    )
    parser.add_argument("--dense-max", type=int, default=2048)
    parser.add_argument("--pot-max", type=int, default=512)
    parser.add_argument(
        "--no-gaussian-reference",
        dest="gaussian_reference",
        action="store_false",
        help="Disable the closed-form population Gaussian W2 reference row.",
    )
    parser.set_defaults(gaussian_reference=True)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    rows = run(args)
    summarize(rows)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
