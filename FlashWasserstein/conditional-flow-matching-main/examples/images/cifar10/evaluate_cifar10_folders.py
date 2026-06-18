"""Offline folder-based evaluation for CIFAR-10 global OT-CFM runs.

This script avoids CleanFID's CIFAR statistics download path by writing a local
CIFAR-10 reference image folder and comparing generated folders against it.
It can evaluate every run directory under ``--run_root`` that contains the
requested ``weights_step_XXXXXXXX.pt`` checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torchvision import datasets


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402


def init_distributed() -> tuple[torch.device, int, int]:
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, 0, 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend)
    return device, rank, world_size


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_max_float(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def parse_csv_ints(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def build_model(train_args: dict, device: torch.device) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=int(train_args.get("num_res_blocks", 2)),
        num_channels=int(train_args.get("num_channel", 128)),
        channel_mult=parse_csv_ints(train_args.get("channel_mult", "1,2,2,2")),
        num_heads=int(train_args.get("num_heads", 4)),
        num_head_channels=int(train_args.get("num_head_channels", 64)),
        attention_resolutions=train_args.get("attention_resolutions", "16"),
        dropout=float(train_args.get("dropout", 0.1)),
    ).to(device)


def load_model(checkpoint_path: Path, device: torch.device, state_key: str) -> UNetModelWrapper:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = build_model(train_args, device)
    if state_key not in checkpoint:
        available = ", ".join(sorted(k for k, v in checkpoint.items() if isinstance(v, dict)))
        raise KeyError(f"{checkpoint_path} has no state key {state_key!r}; available dict keys: {available}")
    state = {k.removeprefix("module."): v for k, v in checkpoint[state_key].items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def integrate(
    model,
    x: torch.Tensor,
    *,
    steps: int,
    method: str,
    amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t = torch.full((x.shape[0],), idx / float(steps), device=x.device)
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp and x.is_cuda):
            if method == "euler":
                x = x + dt * model(t, x).float()
            elif method == "heun":
                v0 = model(t, x).float()
                x_pred = x + dt * v0
                t_next = torch.full((x.shape[0],), (idx + 1) / float(steps), device=x.device)
                v1 = model(t_next, x_pred).float()
                x = x + 0.5 * dt * (v0 + v1)
            else:
                raise ValueError(f"Unknown integration method: {method}")
    return x


def image_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.exists() else 0


def write_cifar_reference(data_dir: Path, out_dir: Path, split: str) -> int:
    expected = 50000 if split == "train" else 10000
    if image_count(out_dir) >= expected:
        return expected
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(root=data_dir, train=(split == "train"), download=True)
    for idx, (image, label) in enumerate(dataset):
        image.save(out_dir / f"{idx:05d}_{int(label):02d}.png")
    return len(dataset)


def save_tensor_images(x: torch.Tensor, out_dir: Path, indices: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    for arr, idx in zip(x, indices, strict=True):
        Image.fromarray(arr).save(out_dir / f"{idx:08d}.png")


@torch.no_grad()
def generate_folder(
    model,
    out_dir: Path,
    *,
    num_gen: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    rank: int,
    world_size: int,
    integration_steps: int,
    integration_method: str,
    amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    if image_count(out_dir) >= num_gen:
        return 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1009 * rank)
    start = time.perf_counter()
    indices = list(range(rank, num_gen, world_size))
    cursor = 0
    while cursor < len(indices):
        batch_indices = indices[cursor : cursor + batch_size]
        cursor += len(batch_indices)
        missing = [idx for idx in batch_indices if not (out_dir / f"{idx:08d}.png").exists()]
        if not missing:
            continue
        batch = len(missing)
        x = torch.randn(batch, 3, 32, 32, generator=generator, device=device)
        x = integrate(
            model,
            x,
            steps=integration_steps,
            method=integration_method,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        save_tensor_images(x, out_dir, missing)
        print(f"rank {rank}: generated through index {missing[-1]} -> {out_dir}", flush=True)
    return time.perf_counter() - start


def compute_folder_scores(gen_dir: Path, real_dir: Path, mode: str, compute_kid: bool) -> dict:
    from cleanfid import fid

    scores = {
        "fid": float(fid.compute_fid(fdir1=str(gen_dir), fdir2=str(real_dir), mode=mode)),
    }
    if compute_kid:
        scores["kid"] = float(fid.compute_kid(fdir1=str(gen_dir), fdir2=str(real_dir), mode=mode))
    return scores


def select_run_dirs(run_root: Path, include: str) -> list[Path]:
    run_dirs = [p for p in sorted(run_root.iterdir()) if p.is_dir()]
    if not include:
        return run_dirs
    needles = [item.strip() for item in include.split(",") if item.strip()]
    return [p for p in run_dirs if any(needle in p.name for needle in needles)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="Directory containing per-method run subdirectories")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_dir", default="~/datasets/cifar10")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--num_gen", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--integration_method", default="euler", choices=["euler", "heun"])
    parser.add_argument("--state_key", default="ema_model", choices=["ema_model", "net_model"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fid_mode", default="clean")
    parser.add_argument("--compute_kid", action="store_true")
    parser.add_argument("--include", default="", help="Comma-separated run-name substrings to evaluate")
    parser.add_argument("--only_generate", action="store_true")
    parser.add_argument("--only_score", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    device, rank, world_size = init_distributed()
    run_root = Path(args.run_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    step_tag = f"{args.step:08d}"
    real_dir = out_dir / f"cifar10_{args.split}_real"
    gen_root = out_dir / f"generated_step_{step_tag}_{args.integration_method}{args.integration_steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"device={device} world_size={world_size}", flush=True)
        print(f"writing/checking CIFAR-10 {args.split} reference: {real_dir}", flush=True)
        n_real = write_cifar_reference(data_dir, real_dir, args.split)
        print(f"reference images={n_real}", flush=True)
    barrier()

    results = []
    for run_dir in select_run_dirs(run_root, args.include):
        checkpoint = run_dir / f"weights_step_{step_tag}.pt"
        if not checkpoint.exists():
            print(f"skip {run_dir.name}: missing {checkpoint.name}", flush=True)
            continue
        gen_dir = gen_root / run_dir.name
        gen_time = None
        if not args.only_score:
            print(f"loading {checkpoint}", flush=True)
            model = load_model(checkpoint, device, args.state_key)
            local_gen_time = generate_folder(
                model,
                gen_dir,
                num_gen=args.num_gen,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
                rank=rank,
                world_size=world_size,
                integration_steps=args.integration_steps,
                integration_method=args.integration_method,
                amp=args.amp,
                amp_dtype=amp_dtype,
            )
            gen_time = reduce_max_float(local_gen_time, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        barrier()

        row = {
            "run": run_dir.name,
            "checkpoint": str(checkpoint),
            "gen_dir": str(gen_dir),
            "real_dir": str(real_dir),
            "step": args.step,
            "num_gen": args.num_gen,
            "split": args.split,
            "integration_method": args.integration_method,
            "integration_steps": args.integration_steps,
            "state_key": args.state_key,
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype if args.amp else "",
            "generation_time_s": gen_time,
        }
        if rank == 0:
            results.append(row)
        barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        scored = []
        for row in results:
            if not args.only_generate:
                gen_dir = Path(row["gen_dir"])
                if image_count(gen_dir) < args.num_gen:
                    raise RuntimeError(f"{gen_dir} has fewer than {args.num_gen} PNGs")
                print(f"scoring {row['run']}", flush=True)
                row.update(compute_folder_scores(gen_dir, real_dir, args.fid_mode, args.compute_kid))
                print(json.dumps(row, indent=2), flush=True)
            scored.append(row)
        out_json = out_dir / f"eval_step_{step_tag}_{args.integration_method}{args.integration_steps}_{args.num_gen}.json"
        out_json.write_text(json.dumps(scored, indent=2))
        print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
