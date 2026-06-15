"""Standard CIFAR-10 pixel-space FM/OT-CFM with local or global couplings.

This is the first benchmark in the Flash global OT-CFM protocol.  It keeps the
standard CIFAR UNet recipe but replaces the pairing stage with the shared
row-conditional coupler from ``torchcfm.ot_coupling``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
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


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


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
def generate_sample_grid(
    model,
    out_dir: Path,
    *,
    step: int,
    device: torch.device,
    sample_batch: int,
    integration_steps: int,
) -> None:
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
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./results_cifar10_global_ot")
    parser.add_argument("--coupling_mode", default="independent", choices=sorted(COUPLING_MODES))
    parser.add_argument("--context_size", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--cost_feature_dim", type=int, default=0, help="0 uses full flattened pixels")
    parser.add_argument("--pot_max_context", type=int, default=2048)
    parser.add_argument("--pot_num_threads", type=parse_num_threads, default=1)
    parser.add_argument("--batch_size", type=int, default=128, help="global batch size under DDP")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--save_step", type=int, default=10000)
    parser.add_argument("--sample_every", type=int, default=5000)
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

    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
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
    out_dir = Path(args.output_dir).expanduser() / run_name
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = sum(p.numel() for p in unwrap(net_model).parameters())
        print(f"dataset=CIFAR10 train images={len(dataset)}")
        print(f"world_size={world_size} global_batch={args.batch_size} local_batch={local_batch}")
        print(f"coupling={args.coupling_mode} context={args.context_size} eps={args.eps}")
        print(f"model_params={params / 1024 / 1024:.2f}M output={out_dir}")
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    metrics_path = out_dir / "metrics.jsonl"
    progress = trange(args.total_steps, dynamic_ncols=True, disable=not is_rank0)
    for step in progress:
        sync_if_cuda(device)
        step_start = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        x1, _labels = next(data_iter)
        x1 = x1.to(device, non_blocking=True)
        x0 = torch.randn_like(x1)

        sync_if_cuda(device)
        ot_start = time.perf_counter()
        coupled = ot_sampler.sample_pairs(x0, x1, step=step)
        sync_if_cuda(device)
        ot_time = time.perf_counter() - ot_start

        t = torch.rand(coupled.x0.shape[0], device=device)
        t_view = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_view) * coupled.x0 + t_view * coupled.x1
        ut = coupled.x1 - coupled.x0
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
