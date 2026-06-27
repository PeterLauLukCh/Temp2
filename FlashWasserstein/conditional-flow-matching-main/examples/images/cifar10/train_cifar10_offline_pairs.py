"""Train CIFAR-10 flow matching from offline coupled noise/data pairs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets
from torchvision.utils import save_image
from tqdm import trange


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402
from torchcfm.ot_coupling import parse_csv_ints, sync_if_cuda  # noqa: E402


DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def dataset_cls(name: str):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}") from exc


def is_distributed() -> bool:
    return int(torch.cuda.device_count() > 1 and "WORLD_SIZE" in __import__("os").environ and __import__("os").environ["WORLD_SIZE"] != "1")


def setup_distributed() -> tuple[int, int, torch.device]:
    import os

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


def reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=op)
    if op == dist.ReduceOp.SUM:
        tensor /= dist.get_world_size()
    return float(tensor.item())


def ema(source, target, decay: float) -> None:
    source_state = unwrap(source).state_dict()
    target_state = target.state_dict()
    for key in source_state.keys():
        target_state[key].data.copy_(target_state[key].data * decay + source_state[key].data * (1 - decay))


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def make_noise_from_seed(seed: int, shape: tuple[int, int, int]) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))


class OfflineCifarPairDataset(Dataset):
    def __init__(self, *, pair_dir: str | Path, data_dir: str | Path, dataset: str = "cifar10"):
        self.pair_dir = Path(pair_dir).expanduser()
        self.metadata = json.loads((self.pair_dir / "metadata.json").read_text())
        cifar_dataset = dataset_cls(dataset)
        ds = cifar_dataset(root=str(data_dir), train=True, download=False)
        self.raw_data = ds.data
        self.data_size = int(self.raw_data.shape[0])
        self.shards = []
        self.lengths = []
        for path in sorted(self.pair_dir.glob("pairs_*.npz")):
            arrays = np.load(path, mmap_mode="r")
            self.shards.append((path, arrays["data_aug_index"], arrays["noise_seed"]))
            self.lengths.append(int(arrays["data_aug_index"].shape[0]))
        if not self.shards:
            raise FileNotFoundError(f"no pairs_*.npz found under {self.pair_dir}")
        self.offsets = np.cumsum([0] + self.lengths)
        self.image_shape = tuple(self.metadata.get("image_shape", [3, 32, 32]))

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _image_from_aug_index(self, aug_index: int) -> torch.Tensor:
        data_index = int(aug_index) % self.data_size
        flip = int(aug_index) >= self.data_size
        image = torch.from_numpy(self.raw_data[data_index]).to(dtype=torch.float32)
        image = image.permute(2, 0, 1).contiguous().div_(255.0)
        if flip:
            image = torch.flip(image, dims=[2])
        return image.mul_(2.0).sub_(1.0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_id = bisect_right(self.offsets, int(index)) - 1
        local = int(index) - int(self.offsets[shard_id])
        _path, data_aug_index, noise_seed = self.shards[shard_id]
        x1 = self._image_from_aug_index(int(data_aug_index[local]))
        x0 = make_noise_from_seed(int(noise_seed[local]), self.image_shape)
        return x0, x1


def build_model(args) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channel,
        channel_mult=parse_csv_ints(args.channel_mult),
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
    )


@torch.no_grad()
def generate_sample_grid(model, out_dir: Path, *, step: int, device: torch.device, sample_batch: int, integration_steps: int):
    was_training = model.training
    model.eval()
    x = torch.randn(sample_batch, 3, 32, 32, device=device)
    dt = 1.0 / float(integration_steps)
    for idx in range(integration_steps):
        t = torch.full((sample_batch,), idx / float(integration_steps), device=device)
        x = x + dt * model(t, x)
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
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--pair_dir", required=True)
    parser.add_argument("--output_dir", default="./results_cifar10_offline_pairs")
    parser.add_argument("--batch_size", type=int, default=128, help="global batch size under DDP")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=400001)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--save_step", type=int, default=50000)
    parser.add_argument("--sample_every", type=int, default=0)
    parser.add_argument("--sample_batch", type=int, default=64)
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_channel", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", default="1,2,2,2")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=64)
    parser.add_argument("--attention_resolutions", default="16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_rank0 = rank == 0
    if args.batch_size % world_size != 0:
        raise ValueError(f"--batch_size={args.batch_size} must be divisible by world_size={world_size}")
    local_batch = args.batch_size // world_size
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset = OfflineCifarPairDataset(pair_dir=args.pair_dir, data_dir=args.data_dir, dataset=args.dataset)
    sampler_ddp = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=local_batch,
        shuffle=sampler_ddp is None,
        sampler=sampler_ddp,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    data_iter = infinite_loader(loader, sampler_ddp)

    net_model = build_model(args).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    if world_size > 1:
        net_model = DistributedDataParallel(net_model, device_ids=[device.index])
    optim = torch.optim.Adam(unwrap(net_model).parameters(), lr=args.lr)

    def lr_multiplier(step):
        warmup = max(args.warmup, 1)
        if step < warmup:
            return min(step + 1, warmup) / float(warmup)
        if args.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup) / float(max(args.total_steps - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    pair_meta = json.loads((Path(args.pair_dir).expanduser() / "metadata.json").read_text())
    run_name = (
        f"offline_sinkhorn_dot_n{pair_meta.get('coupling_size')}_"
        f"eps{pair_meta.get('relative_eps')}_pairs{len(dataset)}_"
        f"bs{args.batch_size}_seed{args.seed}"
    )
    out_dir = Path(args.output_dir).expanduser() / run_name
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = sum(p.numel() for p in unwrap(net_model).parameters())
        print(f"dataset={args.dataset} offline_pairs={len(dataset)} pair_dir={args.pair_dir}")
        print(f"world_size={world_size} global_batch={args.batch_size} local_batch={local_batch}")
        print(f"model_params={params / 1024 / 1024:.2f}M output={out_dir}")
        (out_dir / "args.json").write_text(json.dumps({**vars(args), "pair_metadata": pair_meta}, indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    progress = trange(args.total_steps, dynamic_ncols=True, disable=not is_rank0)
    for step in progress:
        sync_if_cuda(device)
        step_start = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        x0, x1 = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        t = torch.rand(x0.shape[0], device=device)
        t_view = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_view) * x0 + t_view * x1
        ut = x1 - x0

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            vt = net_model(t, xt)
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
        wall_step = reduce_scalar(step_time, device, op=dist.ReduceOp.MAX)
        should_log = step % args.log_step == 0 or step == args.total_steps - 1
        if is_rank0 and should_log:
            images_per_s = args.batch_size / max(wall_step, 1e-12)
            progress.set_postfix(loss=f"{mean_loss:.4g}", img_s=f"{images_per_s:.1f}")
            row = {
                "step": int(step),
                "loss": mean_loss,
                "step_time_s": wall_step,
                "step_s": wall_step,
                "images_per_s": images_per_s,
                "images_s": images_per_s,
                "images/s": images_per_s,
                "lr": float(sched.get_last_lr()[0]),
                "mode": "offline_sinkhorn_dot_pairs",
                "pair_dir": str(args.pair_dir),
                "offline_pairs": int(len(dataset)),
                "coupling_size": int(pair_meta.get("coupling_size", -1)),
                "relative_eps": float(pair_meta.get("relative_eps", float("nan"))),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

        if is_rank0 and args.save_step > 0 and step > 0 and step % args.save_step == 0:
            save_checkpoint(out_dir / f"weights_step_{step:08d}.pt", net_model, ema_model, optim, sched, step, args)

        if is_rank0 and args.sample_every > 0 and step > 0 and step % args.sample_every == 0:
            generate_sample_grid(
                ema_model,
                out_dir,
                step=step,
                device=device,
                sample_batch=args.sample_batch,
                integration_steps=args.integration_steps,
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

