"""HF-Parquet ImageNet pixel-space FM/OT-CFM with local or global couplings.

This is the parquet-dataset counterpart of ``train_imagefolder_global_ot.py``.
It is intended for resized ImageNet Parquet datasets such as
``benjamin-paine/imagenet-1k-256x256``.  Images are decoded on the fly, resized
to ``--image_size``, normalized to ``[-1, 1]``, and paired by the shared
row-conditional OT coupler.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.utils import save_image
from tqdm import trange


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402
from torchcfm.ot_coupling import (  # noqa: E402
    COUPLING_MODES,
    OTCouplingSampler,
    peak_memory_gb,
    parse_csv_ints,
    parse_num_threads,
    sync_if_cuda,
)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, torch.device]:
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), world_size, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def ema(source, target, decay: float) -> None:
    source_state = unwrap(source).state_dict()
    target_state = target.state_dict()
    for key in source_state.keys():
        target_state[key].data.copy_(target_state[key].data * decay + source_state[key].data * (1 - decay))


def reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=op)
    if op == dist.ReduceOp.SUM:
        tensor /= dist.get_world_size()
    return float(tensor.item())


def list_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("train-*.parquet"))
    if not files:
        files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")
    return files


def parquet_num_rows(files: list[Path]) -> int:
    total = 0
    for path in files:
        total += int(pq.ParquetFile(path).metadata.num_rows)
    return total


def infer_label_column(files: list[Path], requested: str) -> str | None:
    names = pq.ParquetFile(files[0]).schema_arrow.names
    if requested:
        if requested not in names:
            raise ValueError(f"requested label column {requested!r} not found; columns={names}")
        return requested
    for candidate in ("label", "labels", "class_id", "class", "target"):
        if candidate in names:
            return candidate
    return None


def image_bytes(image_scalar) -> bytes:
    try:
        data = image_scalar["bytes"].as_py()
    except Exception:
        value = image_scalar.as_py()
        data = value["bytes"] if isinstance(value, dict) else value
    if data is None:
        raise ValueError("image bytes are missing in parquet row")
    return data


def decode_image(image_scalar, image_size: int, hflip: bool, rng: random.Random) -> torch.Tensor:
    with Image.open(io.BytesIO(image_bytes(image_scalar))) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if hflip and rng.random() < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div_(127.5).sub_(1.0)


class ParquetImageNetDataset(IterableDataset):
    """Stream HF ImageNet Parquet shards across DDP ranks and dataloader workers."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        image_size: int,
        hflip: bool,
        label_column: str,
        require_labels: bool,
        seed: int,
        arrow_batch_size: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.files = list_parquet_files(self.data_dir)
        self.num_rows = parquet_num_rows(self.files)
        self.image_size = int(image_size)
        self.hflip = bool(hflip)
        self.label_column = infer_label_column(self.files, label_column)
        if require_labels and self.label_column is None:
            names = pq.ParquetFile(self.files[0]).schema_arrow.names
            raise ValueError(f"class-conditional training requested but no label column was found; columns={names}")
        self.seed = int(seed)
        self.arrow_batch_size = int(arrow_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker = self.rank * num_workers + worker_id
        total_workers = max(1, self.world_size * num_workers)
        rng = random.Random(self.seed + 1009 * self.epoch + 9176 * global_worker)
        files = list(self.files)
        rng.shuffle(files)
        selected = files[global_worker::total_workers]
        if not selected:
            selected = [files[global_worker % len(files)]]

        columns = ["image"]
        if self.label_column is not None:
            columns.append(self.label_column)

        for path in selected:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=self.arrow_batch_size, columns=columns):
                image_col = batch.column("image")
                label_col = batch.column(self.label_column) if self.label_column is not None else None
                for idx in range(len(image_col)):
                    image = decode_image(image_col[idx], self.image_size, self.hflip, rng)
                    label = int(label_col[idx].as_py()) if label_col is not None else -1
                    yield image, torch.tensor(label, dtype=torch.long)


def infinite_loader(loader, dataset=None):
    epoch = 0
    while True:
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def build_model(args) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, args.image_size, args.image_size),
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channel,
        channel_mult=parse_csv_ints(args.channel_mult),
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        class_cond=args.class_conditional,
        num_classes=args.num_classes,
    )


@torch.no_grad()
def generate_sample_grid(
    model,
    out_dir: Path,
    *,
    step: int,
    device: torch.device,
    image_size: int,
    sample_batch: int,
    integration_steps: int,
    class_conditional: bool,
    num_classes: int,
) -> None:
    was_training = model.training
    model.eval()
    x = torch.randn(sample_batch, 3, image_size, image_size, device=device)
    y = None
    if class_conditional:
        y = torch.arange(sample_batch, device=device, dtype=torch.long) % int(num_classes)
    dt = 1.0 / float(integration_steps)
    for idx in range(integration_steps):
        t = torch.full((sample_batch,), idx / float(integration_steps), device=device)
        x = x + dt * (model(t, x, y) if class_conditional else model(t, x))
    image = x.float().clamp(-1, 1).add(1).mul(0.5).cpu()
    save_image(image, out_dir / f"samples_step_{step:08d}.png", nrow=int(math.sqrt(sample_batch)))
    if was_training:
        model.train()


def save_checkpoint(path: Path, net_model, ema_model, optim, sched, step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net_model": unwrap(net_model).state_dict(),
            "ema_model": ema_model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "step": int(step),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing train-*.parquet shards")
    parser.add_argument("--output_dir", default="./results_hf_parquet_global_ot")
    parser.add_argument("--coupling_mode", default="independent", choices=sorted(COUPLING_MODES))
    parser.add_argument("--context_size", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--cost_feature_dim", type=int, default=0, help="0 uses full flattened pixels")
    parser.add_argument("--class_conditional", action="store_true")
    parser.add_argument("--class_aware_coupling", action="store_true")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--label_column", default="", help="Defaults to the first known label column if present")
    parser.add_argument("--pot_max_context", type=int, default=2048)
    parser.add_argument("--pot_num_threads", type=parse_num_threads, default=1)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--no_hflip", action="store_true")
    parser.add_argument("--arrow_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024, help="global batch size under DDP")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=250000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--save_step", type=int, default=25000)
    parser.add_argument("--sample_every", type=int, default=25000)
    parser.add_argument("--sample_batch", type=int, default=64)
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_channel", type=int, default=192)
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--channel_mult", default="1,2,3,4")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=64)
    parser.add_argument("--attention_resolutions", default="8")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_flash_tf32", action="store_true")
    parser.add_argument("--no_flash_autotune", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_rank0 = rank == 0
    if args.batch_size % world_size != 0:
        raise ValueError(f"--batch_size={args.batch_size} must be divisible by world_size={world_size}")
    local_batch = args.batch_size // world_size
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset = ParquetImageNetDataset(
        args.data_dir,
        image_size=args.image_size,
        hflip=not args.no_hflip,
        label_column=args.label_column,
        require_labels=args.class_conditional or args.class_aware_coupling,
        seed=args.seed,
        arrow_batch_size=args.arrow_batch_size,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    data_iter = infinite_loader(loader, dataset)

    ot_sampler = OTCouplingSampler(
        mode=args.coupling_mode,
        context_size=args.context_size,
        eps=args.eps,
        sinkhorn_iters=args.sinkhorn_iters,
        cost_feature_dim=args.cost_feature_dim,
        seed=args.seed,
        pot_max_context=args.pot_max_context,
        pot_num_threads=args.pot_num_threads,
        flash_allow_tf32=not args.no_flash_tf32,
        flash_autotune=not args.no_flash_autotune,
        class_aware=args.class_aware_coupling,
    )

    net_model = build_model(args).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    if world_size > 1:
        net_model = DistributedDataParallel(net_model, device_ids=[device.index])
    optim = torch.optim.Adam(unwrap(net_model).parameters(), lr=args.lr)

    def warmup_lr(step):
        return min(step + 1, args.warmup) / float(max(args.warmup, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    run_name = (
        f"{args.coupling_mode}_ctx{args.context_size}_eps{args.eps:g}_"
        f"it{args.sinkhorn_iters}_bs{args.batch_size}_seed{args.seed}"
    )
    if args.class_conditional:
        run_name = "classcond_" + run_name
    out_dir = Path(args.output_dir).expanduser() / run_name
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = sum(p.numel() for p in unwrap(net_model).parameters())
        print(f"dataset={args.data_dir} rows={len(dataset)} label_column={dataset.label_column}")
        print(f"world_size={world_size} global_batch={args.batch_size} local_batch={local_batch}")
        print(
            f"coupling={args.coupling_mode} context={args.context_size} eps={args.eps} "
            f"class_conditional={args.class_conditional} class_aware={args.class_aware_coupling}"
        )
        print(f"model_params={params / 1024 / 1024:.2f}M output={out_dir}")
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    progress = trange(args.total_steps, dynamic_ncols=True, disable=not is_rank0)
    for step in progress:
        sync_if_cuda(device)
        step_start = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        x1, labels = next(data_iter)
        x1 = x1.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        x0 = torch.randn_like(x1)

        sync_if_cuda(device)
        ot_start = time.perf_counter()
        coupled = ot_sampler.sample_pairs(
            x0,
            x1,
            y1_local=labels if args.class_conditional or args.class_aware_coupling else None,
            step=step,
        )
        sync_if_cuda(device)
        ot_time = time.perf_counter() - ot_start

        t = torch.rand(coupled.x0.shape[0], device=device)
        t_view = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_view) * coupled.x0 + t_view * coupled.x1
        ut = coupled.x1 - coupled.x0
        y_cond = coupled.y1.long() if args.class_conditional else None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            vt = net_model(t, xt, y_cond) if args.class_conditional else net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(unwrap(net_model).parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()
        sched.step()
        ema(net_model, ema_model, args.ema_decay)
        sync_if_cuda(device)
        step_time = time.perf_counter() - step_start

        mean_loss = reduce_scalar(float(loss.item()), device, op=dist.ReduceOp.SUM)
        mean_ot = reduce_scalar(ot_time, device, op=dist.ReduceOp.SUM)
        wall_step = reduce_scalar(step_time, device, op=dist.ReduceOp.MAX)
        if is_rank0 and (step % args.log_step == 0 or step == args.total_steps - 1):
            images_per_s = args.batch_size / max(wall_step, 1e-12)
            progress.set_postfix(loss=f"{mean_loss:.4g}", ot_s=f"{mean_ot:.3f}", img_s=f"{images_per_s:.1f}")
            row = {
                "step": int(step),
                "loss": mean_loss,
                "ot_time_s": mean_ot,
                "step_time_s": wall_step,
                "images_per_s": images_per_s,
                "peak_mem_gb": peak_memory_gb(device),
                "lr": float(sched.get_last_lr()[0]),
                **coupled.metrics,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        if is_rank0 and args.save_step > 0 and step > 0 and step % args.save_step == 0:
            save_checkpoint(out_dir / f"weights_step_{step:08d}.pt", net_model, ema_model, optim, sched, step, args)

        if is_rank0 and args.sample_every > 0 and step > 0 and step % args.sample_every == 0:
            generate_sample_grid(
                ema_model,
                out_dir,
                step=step,
                device=device,
                image_size=args.image_size,
                sample_batch=args.sample_batch,
                integration_steps=args.integration_steps,
                class_conditional=args.class_conditional,
                num_classes=args.num_classes,
            )

    if is_rank0 and args.save_step > 0:
        save_checkpoint(
            out_dir / f"weights_step_{args.total_steps:08d}.pt",
            net_model,
            ema_model,
            optim,
            sched,
            args.total_steps,
            args,
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
