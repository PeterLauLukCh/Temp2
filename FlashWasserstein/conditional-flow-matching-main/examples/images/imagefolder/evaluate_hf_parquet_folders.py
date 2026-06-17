from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_hf_parquet_global_ot as train_mod


def init_distributed() -> tuple[torch.device, int, int, int]:
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, 0, 1, 0

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
    return device, rank, world_size, local_rank


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_rank0(rank: int) -> bool:
    return rank == 0


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def decode_parquet_image(obj, image_size: int) -> Image.Image:
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            img = Image.open(io.BytesIO(obj["bytes"]))
        elif obj.get("path") is not None:
            img = Image.open(obj["path"])
        else:
            raise ValueError(f"Unknown image object keys: {obj.keys()}")
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(obj))
    else:
        raise TypeError(f"Unsupported image object: {type(obj)}")
    return img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)


def build_real_folder(data_dir: Path, real_dir: Path, num_real: int, image_size: int) -> None:
    if count_pngs(real_dir) >= num_real:
        return

    real_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("train-*.parquet"))
    if not files:
        files = sorted((data_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train parquet shards under {data_dir}")

    idx = count_pngs(real_dir)
    for shard in files:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=1024, columns=["image"]):
            for obj in batch.column(0).to_pylist():
                if idx >= num_real:
                    return
                decode_parquet_image(obj, image_size).save(real_dir / f"{idx:08d}.png")
                idx += 1


def load_model(ckpt_path: Path, state_key: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = vars(train_args)
    ns = SimpleNamespace(**train_args)

    model = train_mod.build_model(ns).to(device)
    state = ckpt.get(state_key) or ckpt.get("ema_model") or ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"No usable model state in {ckpt_path}")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ns


def call_model(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None):
    trials = ((x, t, y), (t, x, y), (x, t), (t, x))
    last_error: Exception | None = None
    for args in trials:
        try:
            return model(*args)
        except Exception as exc:  # Different local trainers use different signatures.
            last_error = exc
    raise RuntimeError(f"Could not call model forward; last error: {last_error}")


def save_tensor_images(x: torch.Tensor, out_dir: Path, indices: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    for arr, idx in zip(x, indices, strict=True):
        Image.fromarray(arr).save(out_dir / f"{idx:08d}.png")


def autocast_context(device: torch.device, enabled: bool, dtype: str):
    if device.type != "cuda" or not enabled:
        return torch.autocast("cpu", enabled=False)
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=amp_dtype)


@torch.no_grad()
def generate_shard(
    model,
    ns,
    out_dir: Path,
    num_gen: int,
    batch_size: int,
    steps: int,
    method: str,
    seed: int,
    device: torch.device,
    rank: int,
    world_size: int,
    amp: bool,
    amp_dtype: str,
) -> None:
    per_rank = math.ceil(num_gen / world_size)
    start = rank * per_rank
    end = min(num_gen, start + per_rank)
    if start >= end:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(getattr(ns, "image_size", 64))
    class_conditional = bool(getattr(ns, "class_conditional", False))
    num_classes = int(getattr(ns, "num_classes", 1000))
    generator = torch.Generator(device=device.type).manual_seed(seed + 100003 * rank)
    cursor = start

    while cursor < end:
        batch_indices = list(range(cursor, min(end, cursor + batch_size)))
        missing = [idx for idx in batch_indices if not (out_dir / f"{idx:08d}.png").exists()]
        cursor += len(batch_indices)
        if not missing:
            continue

        bs = len(missing)
        x = torch.randn(bs, 3, image_size, image_size, device=device, generator=generator)
        y = (
            torch.randint(0, num_classes, (bs,), device=device, generator=generator)
            if class_conditional
            else None
        )
        dt = 1.0 / steps

        for k in range(steps):
            t0 = torch.full((bs,), k / steps, device=device)
            with autocast_context(device, amp, amp_dtype):
                v0 = call_model(model, x, t0, y)
                if method == "euler":
                    x = x + dt * v0.float()
                elif method == "heun":
                    xp = x + dt * v0.float()
                    t1 = torch.full((bs,), min((k + 1) / steps, 1.0), device=device)
                    v1 = call_model(model, xp, t1, y)
                    x = x + 0.5 * dt * (v0.float() + v1.float())
                else:
                    raise ValueError(method)

        save_tensor_images(x, out_dir, missing)


def find_checkpoints(run_root: Path, step: int, include: str | None) -> list[Path]:
    ckpts = sorted(run_root.glob(f"*/weights_step_{step:08d}.pt"))
    if include:
        needles = [part for part in include.split(",") if part]
        ckpts = [path for path in ckpts if any(needle in path.parent.name for needle in needles)]
    if not ckpts:
        raise FileNotFoundError(f"No weights_step_{step:08d}.pt under {run_root}")
    return ckpts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--num_gen", type=int, default=50000)
    parser.add_argument("--num_real", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1024, help="Generation batch per GPU/rank.")
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--integration_method", choices=["euler", "heun"], default="euler")
    parser.add_argument("--state_key", default="ema_model")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compute_kid", action="store_true")
    parser.add_argument("--include", default="", help="Comma-separated substrings of run names to evaluate.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--skip_scores", action="store_true")
    args = parser.parse_args()

    device, rank, world_size, local_rank = init_distributed()
    run_root = Path(args.run_root).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    real_dir = out_dir / "imagenet64_train_real"

    if is_rank0(rank):
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "batch_size_per_gpu": args.batch_size,
                    "num_gen": args.num_gen,
                    "integration": f"{args.integration_method}{args.integration_steps}",
                    "amp": args.amp,
                    "amp_dtype": args.amp_dtype,
                },
                indent=2,
            ),
            flush=True,
        )
        build_real_folder(data_dir, real_dir, args.num_real, 64)
    barrier()

    ckpts = find_checkpoints(run_root, args.step, args.include or None)
    results = []
    for ckpt in ckpts:
        run = ckpt.parent.name
        if is_rank0(rank):
            print(f"Evaluating {run}", flush=True)

        model, ns = load_model(ckpt, args.state_key, device)
        gen_dir = out_dir / f"generated_step_{args.step:08d}_{args.integration_method}{args.integration_steps}" / run
        generate_shard(
            model=model,
            ns=ns,
            out_dir=gen_dir,
            num_gen=args.num_gen,
            batch_size=args.batch_size,
            steps=args.integration_steps,
            method=args.integration_method,
            seed=args.seed,
            device=device,
            rank=rank,
            world_size=world_size,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        barrier()

        if is_rank0(rank) and not args.skip_scores:
            from cleanfid import fid

            fid_value = fid.compute_fid(
                str(gen_dir),
                str(real_dir),
                mode="clean",
                batch_size=args.batch_size,
                num_workers=8,
                device=device,
            )
            item = {
                "run": run,
                "checkpoint": str(ckpt),
                "gen_dir": str(gen_dir),
                "real_dir": str(real_dir),
                "step": args.step,
                "num_gen": args.num_gen,
                "num_real": args.num_real,
                "integration_method": args.integration_method,
                "integration_steps": args.integration_steps,
                "state_key": args.state_key,
                "seed": args.seed,
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
                "amp": args.amp,
                "amp_dtype": args.amp_dtype if args.amp else "",
                "fid": float(fid_value),
            }
            if args.compute_kid:
                item["kid"] = float(
                    fid.compute_kid(
                        str(gen_dir),
                        str(real_dir),
                        mode="clean",
                        batch_size=args.batch_size,
                        num_workers=8,
                        device=device,
                    )
                )
            print(json.dumps(item, indent=2), flush=True)
            results.append(item)
        barrier()

    if is_rank0(rank) and not args.skip_scores:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / (
            f"eval_step_{args.step:08d}_{args.integration_method}{args.integration_steps}_{args.num_gen}.json"
        )
        out_json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_json}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
