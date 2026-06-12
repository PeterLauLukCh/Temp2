"""Speed scaling benchmark for FlashWasserstein vs POT exact EMD.

The point of this script is to separate three timings:
  - one Flash hard c-transform oracle call on GPU,
  - the V1 FlashWasserstein semi-dual solver for a fixed number of iterations,
  - POT exact EMD on CPU for sizes where it remains feasible.

POT is intentionally run in a child process with a timeout because exact EMD is
dense and can become impractical quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT / "code" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_gaussian_2d import gaussian_case
from flash_wasserstein import pot_exact_ot, solve_flash_wasserstein


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("This script requires PyTorch.") from exc
    return torch


def maybe_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(torch, device, fn):
    sync(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    sync(torch, device)
    elapsed = time.perf_counter() - start
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return result, elapsed, peak_mb


def repeat_timed(torch, device, fn, *, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
        sync(torch, device)
    seconds = []
    peak = 0.0
    result = None
    for _ in range(repeats):
        result, elapsed, peak_mb = timed(torch, device, fn)
        seconds.append(elapsed)
        peak = max(peak, peak_mb)
    return result, {
        "seconds": seconds,
        "median_seconds": float(statistics.median(seconds)),
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
        "peak_memory_mb": float(peak),
    }


def pot_worker(queue, n: int, case: str, seed: int, cost_scale: float):
    torch = require_torch()
    device = torch.device("cpu")
    x, y = gaussian_case(torch, n, case, seed=seed, device=device)
    start = time.perf_counter()
    result = pot_exact_ot(x, y, cost_scale=cost_scale, max_size=n)
    elapsed = time.perf_counter() - start
    queue.put(
        {
            "seconds": elapsed,
            "transport_cost": result.cost,
            "row_error_l1": result.row_error_l1,
            "col_error_l1": result.col_error_l1,
        }
    )


def run_pot_with_timeout(n: int, case: str, seed: int, cost_scale: float, timeout_s: float):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=pot_worker, args=(queue, n, case, seed, cost_scale))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return {"status": "timeout", "timeout_s": timeout_s}
    if proc.exitcode != 0:
        return {"status": "failed", "exitcode": proc.exitcode}
    if queue.empty():
        return {"status": "failed", "exitcode": proc.exitcode, "message": "empty queue"}
    row = queue.get()
    row["status"] = "ok"
    return row


def flash_oracle_benchmark(torch, device, x, y, *, repeats: int, warmup: int, cost_scale: float):
    from flash_sinkhorn import c_transform_fwd

    psi = torch.zeros((y.shape[0],), device=device, dtype=torch.float32)

    def fn():
        return c_transform_fwd(
            x,
            y,
            psi,
            cost_scale=cost_scale,
            allow_tf32=False,
            autotune=False,
        )

    return repeat_timed(torch, device, fn, repeats=repeats, warmup=warmup)


def flash_solver_benchmark(
    torch,
    device,
    x,
    y,
    *,
    iters: int,
    lr: float,
    repeats: int,
    warmup: int,
    cost_scale: float,
):
    def fn():
        return solve_flash_wasserstein(
            x,
            y,
            cost_scale=cost_scale,
            max_iter=iters,
            lr=lr,
            tol=0.0,
            return_history=False,
            allow_tf32=False,
            autotune=False,
        )

    return repeat_timed(torch, device, fn, repeats=repeats, warmup=warmup)


def add_speed_fields(row: Dict[str, object], n: int, iterations: int, seconds: float):
    interactions = float(n) * float(n) * float(iterations)
    row["interactions"] = interactions
    row["ginteractions_per_s"] = interactions / max(seconds, 1e-12) / 1e9


def write_rows(rows: List[Dict[str, object]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "speed_scaling.json"
    csv_path = output_dir / "speed_scaling.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def plot_rows(rows: List[Dict[str, object]], output_dir: Path):
    plt = maybe_matplotlib()
    if plt is None:
        return None
    method_style = {
        "flash_oracle": ("#0f766e", "o"),
        "flash_solver": ("#2563eb", "s"),
        "pot_emd": ("#f97316", "^"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=180)
    for method, (color, marker) in method_style.items():
        xs = []
        ys = []
        labels = []
        for row in rows:
            if row.get("method") == method and row.get("status", "ok") == "ok":
                xs.append(row["n"])
                ys.append(row["median_seconds"] if "median_seconds" in row else row["seconds"])
                labels.append(row["n"])
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=method)
    for row in rows:
        if row.get("method") == "pot_emd" and row.get("status") == "timeout":
            ax.scatter([row["n"]], [row["timeout_s"]], marker="x", color="#991b1b")
            ax.text(row["n"], row["timeout_s"], " timeout", fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n = m")
    ax.set_ylabel("seconds, log scale")
    ax.set_title("Speed Scaling: Flash GPU vs POT Exact EMD")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "speed_scaling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="shift", choices=["shift", "anisotropic", "near"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-scale", type=float, default=0.5)
    parser.add_argument("--flash-sizes", default="512,1024,2048,8192,20000,50000")
    parser.add_argument("--pot-sizes", default="512,1024,2048,4096")
    parser.add_argument("--solver-iters", type=int, default=200)
    parser.add_argument("--lr-mass-scale", type=float, default=0.05)
    parser.add_argument("--oracle-repeats", type=int, default=10)
    parser.add_argument("--solver-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pot-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "speed_scaling")
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    rows: List[Dict[str, object]] = []
    flash_sizes = parse_csv_ints(args.flash_sizes)
    pot_sizes = parse_csv_ints(args.pot_sizes)

    for n in flash_sizes:
        print(f"\nFlash GPU n={n}")
        x, y = gaussian_case(torch, n, args.case, seed=args.seed, device=device)
        _, oracle_stats = flash_oracle_benchmark(
            torch,
            device,
            x,
            y,
            repeats=args.oracle_repeats,
            warmup=args.warmup,
            cost_scale=args.cost_scale,
        )
        row = {
            "method": "flash_oracle",
            "status": "ok",
            "case": args.case,
            "n": n,
            "iterations": 1,
            **oracle_stats,
        }
        add_speed_fields(row, n, 1, row["median_seconds"])
        print(row)
        rows.append(row)

        lr = args.lr_mass_scale * n
        result, solver_stats = flash_solver_benchmark(
            torch,
            device,
            x,
            y,
            iters=args.solver_iters,
            lr=lr,
            repeats=args.solver_repeats,
            warmup=max(1, min(args.warmup, 2)),
            cost_scale=args.cost_scale,
        )
        row = {
            "method": "flash_solver",
            "status": "ok",
            "case": args.case,
            "n": n,
            "iterations": args.solver_iters,
            "lr": lr,
            "semidual_value": result.semidual_value,
            "transport_cost": result.transport_cost,
            "mass_error_l1": result.mass_error_l1,
            **solver_stats,
        }
        add_speed_fields(row, n, args.solver_iters, row["median_seconds"])
        print(row)
        rows.append(row)

    for n in pot_sizes:
        print(f"\nPOT exact EMD n={n}")
        start = time.perf_counter()
        pot = run_pot_with_timeout(
            n,
            args.case,
            args.seed,
            args.cost_scale,
            args.pot_timeout,
        )
        elapsed = time.perf_counter() - start
        row = {
            "method": "pot_emd",
            "case": args.case,
            "n": n,
            "iterations": 1,
            "seconds": pot.get("seconds", elapsed),
            **pot,
        }
        if row.get("status") == "ok":
            add_speed_fields(row, n, 1, row["seconds"])
        print(row)
        rows.append(row)

    write_rows(rows, args.output_dir)
    plot_path = plot_rows(rows, args.output_dir)
    if plot_path is not None:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
