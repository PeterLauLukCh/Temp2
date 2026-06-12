"""Train OT-CFM on an ImageFolder dataset with optional FlashWasserstein pairing.

Example single GPU:

    CUDA_VISIBLE_DEVICES=0 python train_imagefolder.py \
      --data_root /data/imagenet/train \
      --model otcfm \
      --ot_method flash \
      --batch_size 256 \
      --image_size 64

Example 10 GPU DDP:

    torchrun --standalone --nproc_per_node=10 train_imagefolder.py \
      --data_root /data/imagenet/train \
      --model otcfm \
      --ot_method flash \
      --batch_size 1280 \
      --image_size 64

In DDP mode, OT pairing is local to each rank's minibatch. This increases data
throughput but does not solve one global cross-GPU OT problem.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper


def parse_csv_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v]


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), world_size, torch.device("cuda", local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ema(source, target, decay: float):
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key in source_state.keys():
        target_state[key].data.copy_(target_state[key].data * decay + source_state[key].data * (1 - decay))


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def infinite_loader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_matcher(args):
    sigma = 0.0
    if args.model == "otcfm":
        if args.ot_method == "flash":
            return ExactOptimalTransportConditionalFlowMatcher(
                sigma=sigma,
                ot_method="flash",
                flash_epsilon=args.flash_epsilon,
                flash_epsilon_schedule=parse_csv_floats(args.flash_epsilon_schedule),
                flash_max_rounds=args.flash_max_rounds,
                flash_verify=args.flash_verify,
                flash_fused_bids=not args.no_flash_fused_bids,
                flash_fused_accept=not args.no_flash_fused_accept,
                flash_allow_tf32=not args.no_flash_tf32,
            )
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method=args.ot_method)
    if args.model == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    if args.model == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    if args.model == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    raise ValueError(f"Unknown model {args.model}")


def save_checkpoint(path: Path, net_model, ema_model, optim, sched, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net_model": unwrap(net_model).state_dict(),
            "ema_model": unwrap(ema_model).state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="ImageFolder root, e.g. ImageNet train/")
    parser.add_argument("--output_dir", default="./results_imagefolder")
    parser.add_argument("--model", default="otcfm", choices=["otcfm", "icfm", "fm", "si"])
    parser.add_argument("--ot_method", default="exact", choices=["exact", "sinkhorn", "flash"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256, help="global batch size under DDP")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--save_step", type=int, default=10000)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--num_channel", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", default="1,2,2,2")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=64)
    parser.add_argument("--attention_resolutions", default="16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--flash_epsilon", type=float, default=1e-2)
    parser.add_argument("--flash_epsilon_schedule", default="0.5,0.2,0.1,0.05,0.01")
    parser.add_argument("--flash_max_rounds", type=int, default=200000)
    parser.add_argument("--flash_verify", action="store_true")
    parser.add_argument("--no_flash_fused_bids", action="store_true")
    parser.add_argument("--no_flash_fused_accept", action="store_true")
    parser.add_argument("--no_flash_tf32", action="store_true")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_rank0 = rank == 0
    if args.batch_size % world_size != 0:
        raise ValueError(f"--batch_size={args.batch_size} must be divisible by world_size={world_size}")
    local_batch = args.batch_size // world_size
    if is_rank0 and world_size > 1 and args.model == "otcfm":
        print("DDP note: OT pairing is local to each rank's minibatch, not global across GPUs.")

    dataset = datasets.ImageFolder(args.data_root, transform=build_transform(args.image_size))
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=local_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    data_iter = infinite_loader(loader, sampler)

    net_model = UNetModelWrapper(
        dim=(3, args.image_size, args.image_size),
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channel,
        channel_mult=parse_csv_ints(args.channel_mult),
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
    ).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    if world_size > 1:
        net_model = DistributedDataParallel(net_model, device_ids=[device.index])
        ema_model = DistributedDataParallel(ema_model, device_ids=[device.index])

    optim = torch.optim.Adam(unwrap(net_model).parameters(), lr=args.lr)

    def warmup_lr(step):
        return min(step, args.warmup) / float(args.warmup)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    fm = build_matcher(args)
    run_name = args.model if args.model != "otcfm" or args.ot_method == "exact" else f"{args.model}_{args.ot_method}"
    out_dir = Path(args.output_dir) / run_name

    if is_rank0:
        params = sum(p.numel() for p in unwrap(net_model).parameters())
        print(f"dataset={args.data_root} images={len(dataset)} classes={len(dataset.classes)}")
        print(f"world_size={world_size} global_batch={args.batch_size} local_batch={local_batch}")
        print(f"model_params={params / 1024 / 1024:.2f}M output={out_dir}")

    progress = trange(args.total_steps, dynamic_ncols=True, disable=not is_rank0)
    for step in progress:
        sync_if_cuda(device)
        step_start = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        x1 = next(data_iter)[0].to(device, non_blocking=True)
        x0 = torch.randn_like(x1)

        sync_if_cuda(device)
        ot_start = time.perf_counter()
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        sync_if_cuda(device)
        ot_time = time.perf_counter() - ot_start

        vt = net_model(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap(net_model).parameters(), args.grad_clip)
        optim.step()
        sched.step()
        ema(net_model, ema_model, args.ema_decay)
        sync_if_cuda(device)
        step_time = time.perf_counter() - step_start

        if is_rank0 and (step % args.log_step == 0 or step == args.total_steps - 1):
            progress.set_postfix(loss=f"{loss.item():.4g}", ot_s=f"{ot_time:.3f}", step_s=f"{step_time:.3f}")

        if is_rank0 and args.save_step > 0 and step > 0 and step % args.save_step == 0:
            save_checkpoint(
                out_dir / f"{run_name}_imagefolder_weights_step_{step}.pt",
                net_model,
                ema_model,
                optim,
                sched,
                step,
            )

    if is_rank0 and args.save_step > 0:
        save_checkpoint(
            out_dir / f"{run_name}_imagefolder_weights_step_{args.total_steps}.pt",
            net_model,
            ema_model,
            optim,
            sched,
            args.total_steps,
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
