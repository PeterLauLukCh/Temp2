"""Compare block-local OT and true global OT for OT-CFM minibatches.

The experiment simulates DDP local OT by splitting a global batch into
``num_blocks`` contiguous blocks and solving one independent OT problem per
block. It compares that block-diagonal coupling with a single global OT solve
over the same examples.

This isolates the question: does global OT actually provide lower kinetic
energy / squared velocity than the local minibatch OT used in standard OT-FM?
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import ot as pot
import pyarrow.parquet as pq
import torch
from PIL import Image


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_num_threads(value: int | str) -> int | str:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("num_threads must be positive or 'max'")
        return value
    value = str(value).strip()
    if value == "max":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("num_threads must be positive or 'max'")
    return parsed


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def decode_image(image_struct, image_size: int) -> torch.Tensor:
    data = image_struct["bytes"].as_py()
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div_(127.5).sub_(1.0)


def load_batch(parquet_files: list[Path], batch_size: int, image_size: int) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=min(batch_size, 1024), columns=["image"]):
            image_col = batch.column("image")
            for idx in range(len(image_col)):
                images.append(decode_image(image_col[idx], image_size))
                if len(images) == batch_size:
                    return torch.stack(images, dim=0)
    raise ValueError(f"Only found {len(images)} images across {len(parquet_files)} shard(s).")


def solve_pot_cost(x0: torch.Tensor, x1: torch.Tensor, *, num_threads=1):
    n = x0.shape[0]
    a = pot.unif(n)
    b = pot.unif(n)

    sync_if_cuda(x0.device)
    cost_start = time.perf_counter()
    matrix = torch.cdist(x0.flatten(1), x1.flatten(1)) ** 2
    sync_if_cuda(x0.device)
    cost_time = time.perf_counter() - cost_start

    matrix_np = matrix.detach().cpu().numpy()
    emd_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = pot.emd(a, b, matrix_np, numThreads=num_threads)
    emd_time = time.perf_counter() - emd_start

    mean_cost = float(np.sum(plan * matrix_np))
    assignment = plan.argmax(axis=1)
    unique_targets = int(np.unique(assignment).size)
    assignment_cost = float(matrix_np[np.arange(n), assignment].mean())
    warning_text = "; ".join(str(w.message) for w in caught)
    return {
        "mean_cost": mean_cost,
        "assignment_cost": assignment_cost,
        "cost_time_s": cost_time,
        "emd_time_s": emd_time,
        "total_time_s": cost_time + emd_time,
        "unique_targets": unique_targets,
        "warnings": warning_text,
        "assignment": assignment,
    }


def solve_block_ot(x0: torch.Tensor, x1: torch.Tensor, *, num_blocks: int, num_threads=1):
    n = x0.shape[0]
    if n % num_blocks != 0:
        raise ValueError(f"batch size {n} must be divisible by num_blocks={num_blocks}")
    block = n // num_blocks
    weighted_cost = 0.0
    weighted_assignment_cost = 0.0
    cost_time = 0.0
    emd_time = 0.0
    unique_targets = 0
    warnings_all = []
    assignment = np.empty(n, dtype=np.int64)

    for block_idx in range(num_blocks):
        start = block_idx * block
        end = start + block
        result = solve_pot_cost(x0[start:end], x1[start:end], num_threads=num_threads)
        weight = block / n
        weighted_cost += weight * result["mean_cost"]
        weighted_assignment_cost += weight * result["assignment_cost"]
        cost_time += result["cost_time_s"]
        emd_time += result["emd_time_s"]
        unique_targets += result["unique_targets"]
        assignment[start:end] = result["assignment"] + start
        if result["warnings"]:
            warnings_all.append(f"block{block_idx}: {result['warnings']}")

    return {
        "mean_cost": weighted_cost,
        "assignment_cost": weighted_assignment_cost,
        "cost_time_s": cost_time,
        "emd_time_s": emd_time,
        "total_time_s": cost_time + emd_time,
        "unique_targets": unique_targets,
        "warnings": " | ".join(warnings_all),
        "assignment": assignment,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train-*.parquet shards")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_sizes", default="1280,2560,5120")
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--num_threads", type=parse_num_threads, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="./global_vs_local_ot_bench")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("train-*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for batch_size in parse_csv_ints(args.batch_sizes):
        if batch_size % args.num_blocks != 0:
            raise ValueError(f"batch size {batch_size} must be divisible by {args.num_blocks}")

        load_start = time.perf_counter()
        x1_cpu = load_batch(parquet_files, batch_size, args.image_size)
        load_time = time.perf_counter() - load_start
        x1 = x1_cpu.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)
        sync_if_cuda(device)

        identity_cost = float((x0.float() - x1.float()).flatten(1).pow(2).sum(dim=1).mean().item())
        block_result = solve_block_ot(x0, x1, num_blocks=args.num_blocks, num_threads=args.num_threads)
        global_result = solve_pot_cost(x0, x1, num_threads=args.num_threads)

        local_size = batch_size // args.num_blocks
        global_assignment = global_result["assignment"]
        source_blocks = np.arange(batch_size) // local_size
        target_blocks = global_assignment // local_size
        cross_block_fraction = float(np.mean(source_blocks != target_blocks))

        improvement = block_result["mean_cost"] - global_result["mean_cost"]
        improvement_pct = 100.0 * improvement / block_result["mean_cost"]
        identity_improvement_pct = 100.0 * (identity_cost - global_result["mean_cost"]) / identity_cost

        row = {
            "batch_size": batch_size,
            "num_blocks": args.num_blocks,
            "local_batch": local_size,
            "image_size": args.image_size,
            "dim": int(x1[0].numel()),
            "load_time_s": load_time,
            "identity_cost": identity_cost,
            "block_local_cost": block_result["mean_cost"],
            "global_cost": global_result["mean_cost"],
            "global_vs_block_improvement": improvement,
            "global_vs_block_improvement_pct": improvement_pct,
            "global_vs_identity_improvement_pct": identity_improvement_pct,
            "global_cross_block_fraction": cross_block_fraction,
            "block_total_time_s": block_result["total_time_s"],
            "block_cost_time_s": block_result["cost_time_s"],
            "block_emd_time_s": block_result["emd_time_s"],
            "global_total_time_s": global_result["total_time_s"],
            "global_cost_time_s": global_result["cost_time_s"],
            "global_emd_time_s": global_result["emd_time_s"],
            "block_unique_targets": block_result["unique_targets"],
            "global_unique_targets": global_result["unique_targets"],
            "block_warnings": block_result["warnings"],
            "global_warnings": global_result["warnings"],
        }
        rows.append(row)
        print(row, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "global_vs_local_ot.csv"
    json_path = out_dir / "global_vs_local_ot.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
