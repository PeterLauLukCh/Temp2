"""Benchmark OT pairing on real ImageFolder minibatches.

This measures only the OT section used by OT-CFM, so it is the fastest way to
check whether FlashWasserstein helps for a target image size and batch size.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchcfm.optimal_transport import OTPlanSampler


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


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
    parser.add_argument("--data_root", required=True, help="ImageFolder root")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_sizes", default="128,256,512")
    parser.add_argument("--methods", default="exact,flash")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_pot_batch", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flash_epsilon", type=float, default=1e-2)
    parser.add_argument("--flash_epsilon_schedule", default="0.5,0.2,0.1,0.05,0.01")
    parser.add_argument("--flash_max_rounds", type=int, default=200000)
    parser.add_argument("--out", default="./imagefolder_ot_pairing_bench")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(args.data_root, transform=build_transform(args.image_size))
    rows = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    schedule = parse_csv_floats(args.flash_epsilon_schedule)

    for batch_size in parse_csv_ints(args.batch_sizes):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        x1, _ = next(iter(loader))
        x1 = x1.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)

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
            rows.append(
                {
                    "batch_size": batch_size,
                    "image_size": args.image_size,
                    "dim": int(x1[0].numel()),
                    "method": method,
                    "time_min_s": t_min,
                    "time_mean_s": t_mean,
                    "mean_pair_cost": pair_cost(*pairs),
                }
            )
            print(rows[-1], flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "imagefolder_ot_pairing_bench.csv"
    json_path = out_dir / "imagefolder_ot_pairing_bench.json"
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
