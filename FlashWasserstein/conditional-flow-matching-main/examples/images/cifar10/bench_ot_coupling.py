"""Microbenchmark CIFAR-10 pixel-space OT coupling methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchcfm.ot_coupling import (  # noqa: E402
    FeatureProjector,
    dense_sinkhorn_row_conditional_indices,
    duplicate_fraction,
    flash_sinkhorn_row_conditional_indices,
    pair_cost,
    parse_csv_ints,
    peak_memory_gb,
    pot_row_conditional_indices,
    reset_peak_memory,
    sync_if_cuda,
)


def load_cifar_batch(data_dir: str, n: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=n, shuffle=False, num_workers=4, drop_last=False)
    images, _ = next(iter(loader))
    if images.shape[0] < n:
        raise ValueError(f"requested {n} CIFAR samples but loaded {images.shape[0]}")
    return images.to(device=device, dtype=torch.float32)


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
    elif method == "local_exact_pot" or method == "global_pot_exact_small":
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
    elif method == "local_entropic" or method == "allgather_dense_entropic":
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
    elif method == "flash_global_entropic":
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
    return {
        "method": method,
        "time_s": elapsed,
        "peak_mem_gb": peak_memory_gb(device),
        "sample_cost": float(sample_cost.item()),
        "duplicate_fraction": duplicate_fraction(j, target_h.shape[0]),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--context_sizes", default="512,1024,2048,4096,8192,16384")
    parser.add_argument("--local_batch", type=int, default=128)
    parser.add_argument(
        "--methods",
        default="independent,global_pot_exact_small,allgather_dense_entropic,flash_global_entropic",
    )
    parser.add_argument("--dense_max_context", type=int, default=4096)
    parser.add_argument("--pot_max_context", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--cost_feature_dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pot_num_threads", default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="./cifar10_ot_coupling_bench")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    max_context = max(parse_csv_ints(args.context_sizes))
    target_x = load_cifar_batch(args.data_dir, max_context, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 7)
    source_x = torch.randn(target_x.shape, generator=generator, device=device, dtype=torch.float32)
    projector = FeatureProjector(args.cost_feature_dim, seed=args.seed)
    target_h = projector.project(target_x)
    source_h = projector.project(source_x)
    cost_scale = 1.0 / (2.0 * source_h.shape[1])
    methods = [m for m in args.methods.split(",") if m]
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for context in parse_csv_ints(args.context_sizes):
        if args.local_batch > context:
            raise ValueError("--local_batch must be <= every context size")
        row_ids = torch.arange(args.local_batch, device=device, dtype=torch.long)
        for method in methods:
            if "pot" in method and context > args.pot_max_context:
                rows.append({"context_size": context, "method": method, "skipped": "context exceeds pot_max_context"})
                continue
            if "dense" in method and context > args.dense_max_context:
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
                result = {"method": method, "skipped": f"{type(exc).__name__}: {exc}"}
            result["context_size"] = context
            result["feature_dim"] = source_h.shape[1]
            result["eps"] = args.eps
            result["sinkhorn_iters"] = args.sinkhorn_iters
            rows.append(result)
            print(result, flush=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = out_dir / "cifar10_ot_coupling.csv"
    json_path = out_dir / "cifar10_ot_coupling.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
