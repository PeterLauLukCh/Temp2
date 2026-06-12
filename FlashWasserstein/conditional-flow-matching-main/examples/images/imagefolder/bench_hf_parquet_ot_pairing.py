"""Benchmark OT pairing on Hugging Face ImageNet Parquet shards.

This script is meant for resized ImageNet repos such as
benjamin-paine/imagenet-1k-256x256. It decodes one local Parquet shard into a
minibatch tensor, moves the minibatch to GPU, and times only the OT pairing
section used by OT-CFM.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from torchcfm.optimal_transport import OTPlanSampler


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


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


def pair_cost(x0, x1):
    return float((x0.float() - x1.float()).flatten(1).pow(2).sum(dim=1).mean().item())


def time_sampler(sampler, x0, x1, *, repeat: int, warmup: int, device):
    for _ in range(warmup):
        sampler.sample_plan(x0, x1, replace=False)
    sync_if_cuda(device)
    times = []
    last = None
    for _ in range(repeat):
        sync_if_cuda(device)
        start = time.perf_counter()
        last = sampler.sample_plan(x0, x1, replace=False)
        sync_if_cuda(device)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train-*.parquet shards")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_sizes", default="64,128,256,512")
    parser.add_argument("--methods", default="exact")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max_pot_batch", type=int, default=4096)
    parser.add_argument("--flash_epsilon", type=float, default=1e-2)
    parser.add_argument("--flash_epsilon_schedule", default="0.5,0.2,0.1,0.05,0.01")
    parser.add_argument("--flash_max_rounds", type=int, default=200000)
    parser.add_argument("--out", default="./hf_parquet_ot_pairing_bench")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("train-*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    schedule = parse_csv_floats(args.flash_epsilon_schedule)

    for batch_size in parse_csv_ints(args.batch_sizes):
        x1_cpu = load_batch(parquet_files, batch_size, args.image_size)
        x1 = x1_cpu.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)
        sync_if_cuda(device)

        for method in methods:
            if method == "exact" and batch_size > args.max_pot_batch:
                continue
            if method == "flash":
                sampler = OTPlanSampler(
                    method="flash",
                    flash_epsilon=args.flash_epsilon,
                    flash_epsilon_schedule=schedule,
                    flash_max_rounds=args.flash_max_rounds,
                )
                warmup = args.warmup
            else:
                sampler = OTPlanSampler(method=method)
                warmup = 0

            t_min, t_mean, pairs = time_sampler(
                sampler,
                x0,
                x1,
                repeat=args.repeat,
                warmup=warmup,
                device=device,
            )
            row = {
                "batch_size": batch_size,
                "image_size": args.image_size,
                "dim": int(x1[0].numel()),
                "method": method,
                "time_min_s": t_min,
                "time_mean_s": t_mean,
                "mean_pair_cost": pair_cost(*pairs),
            }
            rows.append(row)
            print(row, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hf_parquet_ot_pairing_bench.csv"
    json_path = out_dir / "hf_parquet_ot_pairing_bench.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "image_size",
                "dim",
                "method",
                "time_min_s",
                "time_mean_s",
                "mean_pair_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
