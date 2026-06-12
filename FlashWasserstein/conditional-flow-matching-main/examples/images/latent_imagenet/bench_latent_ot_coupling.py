"""Microbench latent OT coupling methods on cached ImageNet VAE latents."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_ot import (  # noqa: E402
    LatentProjector,
    dense_sinkhorn_row_conditional_indices,
    duplicate_fraction,
    flash_sinkhorn_row_conditional_indices,
    pair_cost,
    parse_csv_ints,
    pot_row_conditional_indices,
    sync_if_cuda,
)


def load_latents(latent_dir: Path, n: int) -> torch.Tensor:
    shards = sorted(latent_dir.glob("latents_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No latents_*.pt shards found in {latent_dir}")
    chunks = []
    total = 0
    for path in shards:
        z = torch.load(path, map_location="cpu")["latents"]
        need = n - total
        chunks.append(z[:need])
        total += min(int(z.shape[0]), need)
        if total >= n:
            break
    if total < n:
        raise ValueError(f"Requested {n} latents but found only {total}.")
    return torch.cat(chunks, dim=0)[:n].contiguous()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def run_method(
    method: str,
    source_h: torch.Tensor,
    target_h: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    eps: float,
    sinkhorn_iters: int,
    cost_scale: float,
    seed: int,
    step: int,
    pot_num_threads: int | str,
):
    device = source_h.device
    reset_peak_memory(device)
    sync_if_cuda(device)
    start = time.perf_counter()
    extra = {}
    if method == "independent":
        j = row_ids.clone()
    elif method == "global_pot_exact_small":
        j, extra = pot_row_conditional_indices(
            source_h,
            target_h,
            row_ids,
            cost_scale=cost_scale,
            seed=seed,
            step=step,
            rank=0,
            num_threads=pot_num_threads,
        )
    elif method == "global_dense_sinkhorn":
        j, extra = dense_sinkhorn_row_conditional_indices(
            source_h,
            target_h,
            row_ids,
            eps=eps,
            n_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            seed=seed,
            step=step,
        )
    elif method == "global_flash_sinkhorn":
        j, extra = flash_sinkhorn_row_conditional_indices(
            source_h,
            target_h,
            row_ids,
            eps=eps,
            n_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            seed=seed,
            step=step,
        )
    else:
        raise ValueError(method)
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    sample_cost = pair_cost(source_h[row_ids], target_h[j], cost_scale).mean()
    row = {
        "method": method,
        "time_s": elapsed,
        "peak_mem_gb": peak_memory_gb(device),
        "sample_cost": float(sample_cost.item()),
        "duplicate_fraction": duplicate_fraction(j, target_h.shape[0]),
        **extra,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--projection_path", default="")
    parser.add_argument("--context_sizes", default="512,1024,2048,4096,8192,16384")
    parser.add_argument("--local_batch", type=int, default=128)
    parser.add_argument("--methods", default="independent,global_pot_exact_small,global_dense_sinkhorn,global_flash_sinkhorn")
    parser.add_argument("--dense_max_context", type=int, default=4096)
    parser.add_argument("--pot_max_context", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pot_num_threads", default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="./latent_ot_coupling_bench")
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir).expanduser()
    projection_path = Path(args.projection_path).expanduser() if args.projection_path else latent_dir / "projection.pt"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    projector = LatentProjector.load(projection_path, device)
    cost_scale = 1.0 / (2.0 * projector.dim)
    max_context = max(parse_csv_ints(args.context_sizes))
    target_z = load_latents(latent_dir, max_context).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 7)
    source_z = torch.randn(target_z.shape, generator=generator, device=device, dtype=torch.float32)
    target_h = projector.project(target_z)
    source_h = projector.project(source_z)
    methods = [m for m in args.methods.split(",") if m]
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for context in parse_csv_ints(args.context_sizes):
        row_ids = torch.arange(args.local_batch, device=device, dtype=torch.long)
        if args.local_batch > context:
            raise ValueError("--local_batch must be <= every context size")
        for method in methods:
            if method == "global_pot_exact_small" and context > args.pot_max_context:
                rows.append({"context_size": context, "method": method, "skipped": "context exceeds pot_max_context"})
                continue
            if method == "global_dense_sinkhorn" and context > args.dense_max_context:
                rows.append({"context_size": context, "method": method, "skipped": "context exceeds dense_max_context"})
                continue
            try:
                result = run_method(
                    method,
                    source_h[:context],
                    target_h[:context],
                    row_ids,
                    eps=args.eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    cost_scale=cost_scale,
                    seed=args.seed,
                    step=context,
                    pot_num_threads=args.pot_num_threads,
                )
                result["skipped"] = ""
            except Exception as exc:
                result = {
                    "method": method,
                    "skipped": f"{type(exc).__name__}: {exc}",
                }
            result["context_size"] = context
            result["proj_dim"] = projector.dim
            result["eps"] = args.eps
            result["sinkhorn_iters"] = args.sinkhorn_iters
            rows.append(result)
            print(result, flush=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = out_dir / "latent_ot_coupling.csv"
    json_path = out_dir / "latent_ot_coupling.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
