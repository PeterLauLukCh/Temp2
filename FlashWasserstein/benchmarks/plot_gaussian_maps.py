"""Plot Gaussian OT maps learned by FlashWasserstein and POT.

FlashWasserstein returns a deterministic assignment induced by the learned
semi-dual potential. POT returns a Kantorovich plan, so the plotted POT map is
the standard barycentric projection T(x_i) = sum_j pi_ij y_j / a_i.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT / "code" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_gaussian_2d import gaussian_case, gaussian_population_cost
from flash_wasserstein import pot_exact_ot, solve_dense_semidual, solve_flash_wasserstein


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("This script requires PyTorch.") from exc
    return torch


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("This script requires matplotlib.") from exc
    return plt


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
    result = None
    for _ in range(warmup):
        result = fn()
        sync(torch, device)

    seconds = []
    peaks = []
    for _ in range(repeats):
        result, elapsed, peak_mb = timed(torch, device, fn)
        seconds.append(elapsed)
        peaks.append(peak_mb)
    return result, {
        "seconds": seconds,
        "median_seconds": float(statistics.median(seconds)),
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
        "peak_memory_mb": float(max(peaks) if peaks else 0.0),
    }


def barycentric_map(torch, y, plan):
    plan_t = torch.as_tensor(plan, dtype=torch.float32)
    y_cpu = y.detach().cpu().float()
    row_mass = plan_t.sum(dim=1).clamp_min(1e-12)
    return (plan_t @ y_cpu) / row_mass[:, None]


def axis_limits(x_np, y_np, maps):
    all_arrays = [x_np, y_np, *maps]
    mins = [arr.min(axis=0) for arr in all_arrays]
    maxs = [arr.max(axis=0) for arr in all_arrays]
    lo = min(v[0] for v in mins), min(v[1] for v in mins)
    hi = max(v[0] for v in maxs), max(v[1] for v in maxs)
    pad_x = 0.08 * max(hi[0] - lo[0], 1e-6)
    pad_y = 0.08 * max(hi[1] - lo[1], 1e-6)
    return (lo[0] - pad_x, hi[0] + pad_x), (lo[1] - pad_y, hi[1] + pad_y)


def draw_clouds(ax, x_np, y_np, title):
    ax.scatter(x_np[:, 0], x_np[:, 1], s=8, alpha=0.45, c="#2563eb", label="source X")
    ax.scatter(y_np[:, 0], y_np[:, 1], s=8, alpha=0.45, c="#dc2626", label="target Y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def draw_map(ax, x_np, y_np, mapped_np, idx, title, endpoint_label, color):
    ax.scatter(x_np[:, 0], x_np[:, 1], s=7, alpha=0.22, c="#2563eb")
    ax.scatter(y_np[:, 0], y_np[:, 1], s=7, alpha=0.16, c="#dc2626")
    ax.scatter(mapped_np[:, 0], mapped_np[:, 1], s=7, alpha=0.3, c=color, label=endpoint_label)
    starts = x_np[idx]
    ends = mapped_np[idx]
    delta = ends - starts
    ax.quiver(
        starts[:, 0],
        starts[:, 1],
        delta[:, 0],
        delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0024,
        alpha=0.55,
        color=color,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def draw_speed(ax, speed):
    methods = list(speed)
    values = [speed[name]["median_seconds"] for name in methods]
    colors = ["#0f766e", "#7c3aed", "#f97316"][: len(methods)]
    bars = ax.bar(methods, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("median seconds, log scale")
    ax.set_title("Wall-Clock Speed")
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="shift", choices=["shift", "anisotropic", "near"])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-scale", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=16.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--arrows", type=int, default=140)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "maps")
    args = parser.parse_args()

    torch = require_torch()
    plt = require_matplotlib()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y = gaussian_case(torch, args.n, args.case, seed=args.seed, device=device)

    flash_fn = lambda: solve_flash_wasserstein(
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
    flash_result, flash_speed = repeat_timed(
        torch,
        device,
        flash_fn,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    pot_fn = lambda: pot_exact_ot(x, y, cost_scale=args.cost_scale, max_size=args.n)
    pot_result, pot_speed = repeat_timed(
        torch,
        device,
        pot_fn,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    speed = {"FlashWass": flash_speed, "POT EMD": pot_speed}
    dense_result = None
    if args.include_dense:
        dense_fn = lambda: solve_dense_semidual(
            x,
            y,
            cost_scale=args.cost_scale,
            max_iter=args.max_iter,
            lr=args.lr,
            tol=args.tol,
            return_history=False,
        )
        dense_result, dense_speed = repeat_timed(
            torch,
            device,
            dense_fn,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        speed["Dense Semi-Dual"] = dense_speed

    x_cpu = x.detach().cpu().float()
    y_cpu = y.detach().cpu().float()
    flash_map = y_cpu[flash_result.assignment.detach().cpu().long()]
    pot_map = barycentric_map(torch, y, pot_result.plan)

    arrow_count = min(args.arrows, args.n)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 2026)
    arrow_idx = torch.randperm(args.n, generator=generator)[:arrow_count].numpy()

    x_np = x_cpu.numpy()
    y_np = y_cpu.numpy()
    flash_np = flash_map.numpy()
    pot_np = pot_map.numpy()
    xlim, ylim = axis_limits(x_np, y_np, [flash_np, pot_np])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.0), dpi=180)
    draw_clouds(axes[0, 0], x_np, y_np, f"2D Gaussian Clouds: {args.case}, n={args.n}")
    draw_map(
        axes[0, 1],
        x_np,
        y_np,
        flash_np,
        arrow_idx,
        f"FlashWass Deterministic Map\nmass L1={flash_result.mass_error_l1:.3g}, cost={flash_result.transport_cost:.4g}",
        "assigned y_j",
        "#0f766e",
    )
    draw_map(
        axes[1, 0],
        x_np,
        y_np,
        pot_np,
        arrow_idx,
        f"POT Exact Plan, Barycentric Map\ncost={pot_result.cost:.4g}",
        "barycentric T(x)",
        "#7c3aed",
    )
    draw_speed(axes[1, 1], speed)

    for ax in axes.ravel()[:3]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.15)

    gaussian_cost = gaussian_population_cost(torch, args.case, cost_scale=args.cost_scale)
    fig.suptitle(
        (
            f"FlashWasserstein vs POT on Gaussian OT | "
            f"Flash semi-dual={flash_result.semidual_value:.5g}, "
            f"POT empirical={pot_result.cost:.5g}, "
            f"population Gaussian={gaussian_cost:.5g}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    stem = f"gaussian_{args.case}_n{args.n}_seed{args.seed}"
    png_path = args.output_dir / f"{stem}_maps_speed.png"
    json_path = args.output_dir / f"{stem}_maps_speed.json"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": args.case,
        "n": args.n,
        "seed": args.seed,
        "cost_scale": args.cost_scale,
        "max_iter": args.max_iter,
        "lr": args.lr,
        "flash": {
            "semidual_value": flash_result.semidual_value,
            "transport_cost": flash_result.transport_cost,
            "mass_error_l1": flash_result.mass_error_l1,
            "n_iter": flash_result.n_iter,
            "converged": flash_result.converged,
            "unique_assigned_targets": int(torch.unique(flash_result.assignment).numel()),
        },
        "pot": {
            "transport_cost": pot_result.cost,
            "row_error_l1": pot_result.row_error_l1,
            "col_error_l1": pot_result.col_error_l1,
        },
        "gaussian_population_cost": gaussian_cost,
        "speed": speed,
        "dense": None
        if dense_result is None
        else {
            "semidual_value": dense_result.semidual_value,
            "transport_cost": dense_result.transport_cost,
            "mass_error_l1": dense_result.mass_error_l1,
        },
        "plot": str(png_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
