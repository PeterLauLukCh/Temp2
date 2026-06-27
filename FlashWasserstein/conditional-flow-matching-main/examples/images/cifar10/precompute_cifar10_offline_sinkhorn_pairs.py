"""Precompute CIFAR-10 offline Sinkhorn noise/data pairs.

This implements the offline-pair construction from
"On Fitting Flow Models with Large Sinkhorn Couplings":

  * build an OT batch of Gaussian noise and CIFAR images,
  * use the negative dot-product cost C = -X Y^T,
  * pass eps_tilde = std(C) * relative_eps to Sinkhorn,
  * sample one target index per source row from the entropic coupling,
  * store only (augmented data index, noise PRNG seed).

For CIFAR-10 random horizontal flips, the augmented data identifier is encoded
as ``data_index + flip * len(dataset)``. Training can exactly reconstruct the
same image and noise vector from this compact pair record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from tqdm import trange


DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def dataset_cls(name: str):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}") from exc


def augmented_cifar_batch(raw_data: np.ndarray, augmented_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return normalized [-1, 1] CIFAR tensors for augmented ids."""

    n_data = int(raw_data.shape[0])
    data_indices = np.asarray(augmented_indices % n_data, dtype=np.int64)
    flip = np.asarray(augmented_indices >= n_data, dtype=np.bool_)
    images = torch.from_numpy(raw_data[data_indices]).to(device=device, dtype=torch.float32)
    images = images.permute(0, 3, 1, 2).contiguous().div_(255.0)
    if bool(flip.any()):
        flip_t = torch.from_numpy(flip).to(device=device)
        images[flip_t] = torch.flip(images[flip_t], dims=[3])
    return images.mul_(2.0).sub_(1.0)


def make_noise_from_seeds_cpu(seeds: np.ndarray, shape: tuple[int, int, int]) -> torch.Tensor:
    """Regenerate per-example Gaussian noise from stored PRNG seeds.

    This is intentionally simple and deterministic. It mirrors the paper's
    "store PRNG key, not noise vector" design; both precompute and training use
    the same NumPy generator convention.
    """

    out = np.empty((len(seeds), *shape), dtype=np.float32)
    for i, seed in enumerate(np.asarray(seeds, dtype=np.int64)):
        rng = np.random.default_rng(int(seed))
        out[i] = rng.standard_normal(shape, dtype=np.float32)
    return torch.from_numpy(out)


class AugmentedIndexSampler:
    """Shuffle augmented CIFAR ids; refresh as epochs are exhausted."""

    def __init__(self, augmented_size: int, seed: int, replacement: bool):
        self.augmented_size = int(augmented_size)
        self.replacement = bool(replacement)
        self.rng = np.random.default_rng(int(seed))
        self.perm = np.empty((0,), dtype=np.int64)
        self.offset = 0

    def sample(self, n: int) -> np.ndarray:
        n = int(n)
        if self.replacement:
            return self.rng.integers(0, self.augmented_size, size=n, dtype=np.int64)

        chunks = []
        remaining = n
        while remaining > 0:
            if self.offset >= len(self.perm):
                self.perm = self.rng.permutation(self.augmented_size).astype(np.int64, copy=False)
                self.offset = 0
            take = min(remaining, len(self.perm) - self.offset)
            chunks.append(self.perm[self.offset : self.offset + take])
            self.offset += take
            remaining -= take
        return np.concatenate(chunks, axis=0)


@torch.no_grad()
def sinkhorn_dot_cost(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    relative_eps: float,
    max_iters: int,
    threshold: float,
    check_every: int,
    sample_chunk: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Run balanced log-domain Sinkhorn with paper-style dot-product cost."""

    x = x0.flatten(1).float().contiguous()
    y = x1.flatten(1).float().contiguous()
    n = int(x.shape[0])
    if int(y.shape[0]) != n:
        raise ValueError(f"expected square OT batch, got {x.shape[0]} and {y.shape[0]}")

    t0 = time.perf_counter()
    cost = -(x @ y.t())
    cost_std = float(cost.std(unbiased=False).item())
    eps_tilde = max(float(relative_eps) * max(cost_std, 1e-12), 1e-12)
    log_k = -cost / eps_tilde
    log_a = -math.log(n)
    log_b = -math.log(n)
    u = torch.zeros(n, device=x.device, dtype=torch.float32)
    v = torch.zeros(n, device=x.device, dtype=torch.float32)
    solve_start = time.perf_counter()
    marginal_l1 = float("inf")
    iterations = 0

    for it in range(1, int(max_iters) + 1):
        u = log_a - torch.logsumexp(log_k + v.view(1, -1), dim=1)
        v = log_b - torch.logsumexp(log_k.t() + u.view(1, -1), dim=1)
        iterations = it
        if it == 1 or it % int(check_every) == 0:
            row_sum = torch.empty(n, device=x.device, dtype=torch.float32)
            col_sum = torch.zeros(n, device=x.device, dtype=torch.float32)
            for start in range(0, n, int(sample_chunk)):
                stop = min(start + int(sample_chunk), n)
                log_p = log_k[start:stop] + u[start:stop].view(-1, 1) + v.view(1, -1)
                p = torch.exp(log_p)
                row_sum[start:stop] = p.sum(dim=1)
                col_sum += p.sum(dim=0)
            marginal_l1 = float((row_sum - 1.0 / n).abs().sum().add((col_sum - 1.0 / n).abs().sum()).item())
            if marginal_l1 <= float(threshold):
                break

    solve_time = time.perf_counter() - solve_start

    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    sampled_cols = []
    sample_cost_sum = 0.0
    duplicate_cols = []
    entropy_sum = 0.0
    for start in range(0, n, int(sample_chunk)):
        stop = min(start + int(sample_chunk), n)
        logits = log_k[start:stop] + v.view(1, -1)
        probs = torch.softmax(logits, dim=1)
        cols = torch.multinomial(probs, 1, replacement=True, generator=generator).flatten()
        sampled_cols.append(cols)
        duplicate_cols.append(cols)
        sample_cost_sum += float(cost[start:stop, cols].sum().item())

        log_p = log_k[start:stop] + u[start:stop].view(-1, 1) + v.view(1, -1)
        p = torch.exp(log_p)
        entropy_sum += float((-(p * log_p)).sum().item())

    j = torch.cat(sampled_cols, dim=0)
    unique = int(torch.unique(torch.cat(duplicate_cols)).numel())
    renorm_entropy = entropy_sum / math.log(n) - 1.0
    metrics = {
        "n": n,
        "relative_eps": float(relative_eps),
        "eps_tilde": eps_tilde,
        "cost_std": cost_std,
        "cost_mode": "negative_dot_product",
        "iterations": int(iterations),
        "converged": bool(marginal_l1 <= float(threshold)),
        "marginal_l1": marginal_l1,
        "threshold": float(threshold),
        "sinkhorn_time_s": solve_time,
        "total_time_s": time.perf_counter() - t0,
        "sample_cost": sample_cost_sum / float(n),
        "duplicate_fraction": 1.0 - unique / float(n),
        "renormalized_entropy": float(renorm_entropy),
    }
    return j, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_pairs", type=int, default=1_000_000)
    parser.add_argument("--coupling_size", type=int, default=2048)
    parser.add_argument("--relative_eps", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=50_000)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--check_every", type=int, default=100)
    parser.add_argument("--sample_chunk", type=int, default=512)
    parser.add_argument("--shard_pairs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_seed", type=int, default=1_000_000_000)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cifar_dataset = dataset_cls(args.dataset)
    ds = cifar_dataset(root=args.data_dir, train=True, download=True)
    raw_data = ds.data
    n_data = int(raw_data.shape[0])
    augmented_size = n_data * 2

    meta = {
        **vars(args),
        "data_size": n_data,
        "augmented_data_size": augmented_size,
        "image_shape": [3, 32, 32],
        "pair_record": "data_aug_index:int32, noise_seed:int64",
        "paper_method": {
            "cost": "negative_dot_product",
            "eps_tilde": "std(C) * relative_eps",
            "threshold_l1": args.threshold,
            "max_iters": args.max_iters,
            "offline_storage": "data identifier plus PRNG seed",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    index_sampler = AugmentedIndexSampler(augmented_size, seed=args.seed + 17, replacement=args.replacement)
    noise_rng = np.random.default_rng(int(args.noise_seed))
    metrics_path = out_dir / "metrics.jsonl"

    shard_data: list[np.ndarray] = []
    shard_seed: list[np.ndarray] = []
    shard_id = 0
    written = 0
    total_blocks = math.ceil(args.num_pairs / args.coupling_size)
    progress = trange(total_blocks, dynamic_ncols=True)

    def flush() -> None:
        nonlocal shard_id, written, shard_data, shard_seed
        if not shard_data:
            return
        data_index = np.concatenate(shard_data, axis=0).astype(np.int32, copy=False)
        noise_seed = np.concatenate(shard_seed, axis=0).astype(np.int64, copy=False)
        path = out_dir / f"pairs_{shard_id:05d}.npz"
        np.savez_compressed(path, data_aug_index=data_index, noise_seed=noise_seed)
        written += int(data_index.shape[0])
        print(f"wrote {path} pairs={data_index.shape[0]} total={written}", flush=True)
        shard_id += 1
        shard_data = []
        shard_seed = []

    for block in progress:
        remaining = args.num_pairs - written - sum(arr.shape[0] for arr in shard_data)
        if remaining <= 0:
            break
        n = min(int(args.coupling_size), int(remaining))
        if n < 2:
            break
        aug_indices = index_sampler.sample(n)
        noise_seeds = noise_rng.integers(0, np.iinfo(np.int64).max, size=n, dtype=np.int64)

        x1 = augmented_cifar_batch(raw_data, aug_indices, device)
        x0 = make_noise_from_seeds_cpu(noise_seeds, (3, 32, 32)).to(device=device, non_blocking=True)

        cols, metrics = sinkhorn_dot_cost(
            x0,
            x1,
            relative_eps=args.relative_eps,
            max_iters=args.max_iters,
            threshold=args.threshold,
            check_every=args.check_every,
            sample_chunk=args.sample_chunk,
            seed=args.seed + 104729 * int(block),
        )
        paired_aug_indices = aug_indices[cols.detach().cpu().numpy()]
        shard_data.append(paired_aug_indices.astype(np.int32, copy=False))
        shard_seed.append(noise_seeds.astype(np.int64, copy=False))

        row = {
            "block": int(block),
            "pairs_total_after_block": written + sum(arr.shape[0] for arr in shard_data),
            **metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        progress.set_postfix(
            it=metrics["iterations"],
            err=f"{metrics['marginal_l1']:.2g}",
            ent=f"{metrics['renormalized_entropy']:.3f}",
            dup=f"{metrics['duplicate_fraction']:.3f}",
        )

        if sum(arr.shape[0] for arr in shard_data) >= int(args.shard_pairs):
            flush()

    flush()
    meta["num_pairs_written"] = written
    meta["num_shards"] = shard_id
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done: wrote {written} pairs to {out_dir}")


if __name__ == "__main__":
    main()

