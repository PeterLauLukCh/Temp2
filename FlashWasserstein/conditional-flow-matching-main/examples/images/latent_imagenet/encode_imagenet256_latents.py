"""Encode HF ImageNet-256 parquet shards into SD-VAE latent shards.

The output directory contains:
  - latents_000000.pt, ... with {"latents": fp16 [N,4,32,32], "labels": long}
  - projection.pt with random projection and calibration statistics
  - metadata.json with command/data metadata
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def extract_image_bytes(image_scalar) -> bytes:
    try:
        return image_scalar["bytes"].as_py()
    except Exception:
        value = image_scalar.as_py()
        if isinstance(value, dict) and "bytes" in value:
            return value["bytes"]
        raise


def decode_image(image_scalar, image_size: int) -> torch.Tensor:
    data = extract_image_bytes(image_scalar)
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)
    return tensor


def find_parquet_files(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")
    return files


def iter_parquet_batches(
    parquet_files: list[Path],
    *,
    batch_size: int,
    image_size: int,
):
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema.names
        wanted = ["image"]
        if "label" in columns:
            wanted.append("label")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=wanted):
            image_col = batch.column("image")
            images = [decode_image(image_col[idx], image_size) for idx in range(len(image_col))]
            labels = None
            if "label" in wanted:
                labels = torch.tensor(batch.column("label").to_pylist(), dtype=torch.long)
            else:
                labels = torch.full((len(images),), -1, dtype=torch.long)
            yield torch.stack(images, dim=0), labels


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor, *, device: torch.device, scaling_factor: float) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            latents = vae.encode(images).latent_dist.mean
    else:
        latents = vae.encode(images).latent_dist.mean
    return latents.mul(float(scaling_factor)).detach().cpu().to(torch.float16)


def flush_shard(
    out_dir: Path,
    shard_prefix: str,
    shard_idx: int,
    latents: list[torch.Tensor],
    labels: list[torch.Tensor],
    *,
    start_index: int,
) -> tuple[Path, int]:
    z = torch.cat(latents, dim=0).contiguous()
    y = torch.cat(labels, dim=0).contiguous()
    path = out_dir / f"{shard_prefix}_{shard_idx:06d}.pt"
    torch.save({"latents": z, "labels": y, "start_index": int(start_index)}, path)
    return path, int(z.shape[0])


def make_projection(flat_dim: int, proj_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    proj = torch.randn((proj_dim, flat_dim), generator=generator, dtype=torch.float32)
    proj.mul_(1.0 / math.sqrt(flat_dim))
    return proj


def update_stats(sum_: torch.Tensor, sumsq: torch.Tensor, count: int, h: torch.Tensor):
    h = h.float()
    return sum_ + h.sum(dim=0).cpu(), sumsq + h.pow(2).sum(dim=0).cpu(), count + h.shape[0]


@torch.no_grad()
def calibrate_projection(
    out_dir: Path,
    shard_paths: list[Path],
    *,
    proj_dim: int,
    seed: int,
    max_samples: int,
    device: torch.device,
) -> Path:
    first = torch.load(shard_paths[0], map_location="cpu")["latents"]
    flat_dim = int(first[0].numel())
    proj = make_projection(flat_dim, proj_dim, seed)
    proj_device = proj.to(device)
    sum_ = torch.zeros(proj_dim, dtype=torch.float64)
    sumsq = torch.zeros(proj_dim, dtype=torch.float64)
    count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 12345)

    for path in shard_paths:
        payload = torch.load(path, map_location="cpu")
        z = payload["latents"]
        if count >= max_samples * 2:
            break
        remaining_targets = max(0, max_samples - count // 2)
        if remaining_targets <= 0:
            break
        z = z[:remaining_targets].to(device=device, dtype=torch.float32)
        flat = z.flatten(1)
        h_target = flat @ proj_device.t()
        z_source = torch.randn(
            z.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        h_source = z_source.flatten(1) @ proj_device.t()
        h = torch.cat([h_target, h_source], dim=0)
        sum_, sumsq, count = update_stats(sum_, sumsq, count, h)

    if count == 0:
        raise RuntimeError("projection calibration saw zero samples.")
    mean = (sum_ / count).float()
    var = (sumsq / count - mean.double().pow(2)).clamp_min(1e-12)
    std = var.sqrt().float().clamp_min(1e-6)
    payload = {
        "proj": proj,
        "mean": mean,
        "std": std,
        "flat_dim": flat_dim,
        "proj_dim": proj_dim,
        "seed": int(seed),
        "calibration_count": int(count),
        "note": "statistics computed over VAE target latents and Gaussian source latents",
    }
    path = out_dir / "projection.pt"
    torch.save(payload, path)
    return path


@torch.no_grad()
def save_reconstruction_grid(
    vae,
    shard_path: Path,
    out_dir: Path,
    *,
    device: torch.device,
    scaling_factor: float,
    n: int = 16,
) -> None:
    z = torch.load(shard_path, map_location="cpu")["latents"][:n].to(device=device, dtype=torch.float32)
    z = z / float(scaling_factor)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image = vae.decode(z).sample
    else:
        image = vae.decode(z).sample
    image = image.float().clamp(-1, 1).add(1).mul(0.5).cpu()
    save_image(image, out_dir / "vae_reconstruction_grid.png", nrow=int(math.sqrt(n)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train-*.parquet shards")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--max_images", type=int, default=0, help="0 means encode all images")
    parser.add_argument("--scaling_factor", type=float, default=0.18215)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--projection_seed", type=int, default=0)
    parser.add_argument("--calibration_samples", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode_rank", type=int, default=0)
    parser.add_argument("--encode_world_size", type=int, default=1)
    parser.add_argument("--shard_prefix", default="")
    parser.add_argument("--skip_projection", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--progress_every", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser()
    parquet_files = find_parquet_files(data_dir, args.split)
    if args.encode_world_size <= 0:
        raise ValueError("--encode_world_size must be positive")
    if not (0 <= args.encode_rank < args.encode_world_size):
        raise ValueError("--encode_rank must be in [0, encode_world_size)")
    if args.encode_world_size > 1:
        parquet_files = [
            path for idx, path in enumerate(parquet_files) if idx % args.encode_world_size == args.encode_rank
        ]
        if not parquet_files:
            raise RuntimeError(f"rank {args.encode_rank} received zero parquet shards")
    shard_prefix = args.shard_prefix or (
        f"latents_r{args.encode_rank:02d}" if args.encode_world_size > 1 else "latents"
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(args.vae_model, local_files_only=args.local_files_only).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    print(
        f"rank={args.encode_rank}/{args.encode_world_size} files={len(parquet_files)} "
        f"device={device} prefix={shard_prefix}",
        flush=True,
    )

    shard_paths: list[Path] = []
    current_latents: list[torch.Tensor] = []
    current_labels: list[torch.Tensor] = []
    shard_idx = 0
    total = 0
    shard_start = 0
    start_time = time.perf_counter()
    next_progress = args.progress_every
    for images, labels in iter_parquet_batches(
        parquet_files,
        batch_size=args.batch_size,
        image_size=args.image_size,
    ):
        if args.max_images > 0:
            remaining = args.max_images - total
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]
        latents = encode_batch(vae, images, device=device, scaling_factor=args.scaling_factor)
        current_latents.append(latents)
        current_labels.append(labels.cpu())
        total += int(latents.shape[0])
        if args.progress_every > 0 and total >= next_progress:
            elapsed = time.perf_counter() - start_time
            print(f"rank={args.encode_rank} encoded={total} elapsed_s={elapsed:.1f}", flush=True)
            while next_progress <= total:
                next_progress += args.progress_every

        buffered = sum(t.shape[0] for t in current_latents)
        if buffered >= args.shard_size:
            path, count = flush_shard(
                out_dir,
                shard_prefix,
                shard_idx,
                current_latents,
                current_labels,
                start_index=shard_start,
            )
            shard_paths.append(path)
            print(f"wrote {path.name} count={count} total={total}", flush=True)
            shard_idx += 1
            shard_start += count
            current_latents = []
            current_labels = []

    if current_latents:
        path, count = flush_shard(
            out_dir,
            shard_prefix,
            shard_idx,
            current_latents,
            current_labels,
            start_index=shard_start,
        )
        shard_paths.append(path)
        print(f"wrote {path.name} count={count} total={total}", flush=True)

    if not shard_paths:
        raise RuntimeError("no latent shards were written.")

    projection_path = None
    if not args.skip_projection:
        projection_path = calibrate_projection(
            out_dir,
            shard_paths,
            proj_dim=args.proj_dim,
            seed=args.projection_seed,
            max_samples=args.calibration_samples,
            device=device,
        )
        save_reconstruction_grid(
            vae,
            shard_paths[0],
            out_dir,
            device=device,
            scaling_factor=args.scaling_factor,
        )
    elapsed = time.perf_counter() - start_time
    metadata = {
        "data_dir": str(data_dir),
        "split": args.split,
        "vae_model": args.vae_model,
        "image_size": args.image_size,
        "scaling_factor": args.scaling_factor,
        "total_images": int(total),
        "num_shards": len(shard_paths),
        "shard_size": args.shard_size,
        "projection": str(projection_path.name) if projection_path is not None else None,
        "encode_rank": args.encode_rank,
        "encode_world_size": args.encode_world_size,
        "shard_prefix": shard_prefix,
        "elapsed_s": elapsed,
    }
    metadata_name = "metadata.json" if args.encode_world_size == 1 else f"metadata_r{args.encode_rank:02d}.json"
    (out_dir / metadata_name).write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
