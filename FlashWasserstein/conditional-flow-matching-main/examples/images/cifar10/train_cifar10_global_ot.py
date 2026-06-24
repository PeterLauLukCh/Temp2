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

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher  # noqa: E402
from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402
from torchcfm.ot_coupling import (  # noqa: E402
    COUPLING_MODES,
    OTCouplingSampler,
    peak_memory_gb,
    parse_csv_ints,
    parse_num_threads,
    sync_if_cuda,
)


OFFICIAL_OTCFM_MODE = "official_otcfm_exact"
TRAINING_MODES = sorted(set(COUPLING_MODES) | {OFFICIAL_OTCFM_MODE})


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


def reduce_sum(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
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


def build_official_otcfm(args) -> ExactOptimalTransportConditionalFlowMatcher:
    return ExactOptimalTransportConditionalFlowMatcher(
        sigma=0.0,
        ot_method="exact",
        num_threads=args.pot_num_threads,
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


@torch.no_grad()
def evaluate_validation_loss(
    model,
    val_iter,
    ot_sampler: OTCouplingSampler | None,
    device: torch.device,
    *,
    val_batches: int,
    step: int,
    seed: int,
    amp: bool,
    official_fm: ExactOptimalTransportConditionalFlowMatcher | None = None,
) -> tuple[float, float, int, int]:
    was_training = model.training
    model.eval()
    total_loss_sum = 0.0
    total_ot_time = 0.0
    total_samples = 0
    total_batches = 0
    batches = max(int(val_batches), 1)
    generator = torch.Generator(device=device)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    generator.manual_seed(int(seed) + 1000003 * int(step) + 9176 * int(rank))

    for val_idx in range(batches):
        x1, _labels = next(val_iter)
        x1 = x1.to(device, non_blocking=True)
        batch_size = int(x1.shape[0])
        x0 = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=generator)

        sync_if_cuda(device)
        ot_start = time.perf_counter()
        if official_fm is not None:
            t, xt, ut = official_fm.sample_location_and_conditional_flow(x0, x1)
        else:
            if ot_sampler is None:
                raise RuntimeError("ot_sampler is required unless official_fm is provided.")
            coupled = ot_sampler.sample_pairs(x0, x1, step=step * batches + val_idx)
            t = torch.rand(coupled.x0.shape[0], device=device, generator=generator)
            t_view = t.view(-1, 1, 1, 1)
            xt = (1.0 - t_view) * coupled.x0 + t_view * coupled.x1
            ut = coupled.x1 - coupled.x0
        sync_if_cuda(device)
        total_ot_time += time.perf_counter() - ot_start

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
        total_loss_sum += float(loss.item()) * batch_size
        total_samples += batch_size
        total_batches += 1

    if was_training:
        model.train()
    return total_loss_sum, total_ot_time, total_samples, total_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./results_cifar10_global_ot")
    parser.add_argument("--coupling_mode", default="independent", choices=TRAINING_MODES)
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
    parser.add_argument("--val_every", type=int, default=0, help="0 disables held-out CIFAR-10 loss")
    parser.add_argument("--val_batches", type=int, default=8, help="0 uses the full validation loader")
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

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if is_rank0:
        datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        if args.val_every > 0:
            datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transform,
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

    val_iter = None
    val_ot_sampler = None
    val_num_batches = 0
    if args.val_every > 0:
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        val_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=False,
            transform=val_transform,
        )
        val_sampler_ddp = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=local_batch,
            shuffle=False,
            sampler=val_sampler_ddp,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_num_batches = len(val_loader)
        val_iter = infinite_loader(val_loader, val_sampler_ddp)

    official_fm = build_official_otcfm(args) if args.coupling_mode == OFFICIAL_OTCFM_MODE else None
    ot_sampler = None
    if official_fm is None:
        ot_sampler = OTCouplingSampler(
            mode=args.coupling_mode,
            context_size=args.context_size,
            eps=args.eps,
            sinkhorn_iters=args.sinkhorn_iters,
            cost_feature_dim=args.cost_feature_dim,
            seed=args.seed + 12345,
            pot_max_context=args.pot_max_context,
            pot_num_threads=args.pot_num_threads,
            flash_allow_tf32=not args.no_flash_tf32,
            flash_autotune=not args.no_flash_autotune,
        )
    if args.val_every > 0 and official_fm is None:
        val_ot_sampler = OTCouplingSampler(
            mode=args.coupling_mode,
            context_size=args.context_size,
            eps=args.eps,
            sinkhorn_iters=args.sinkhorn_iters,
            cost_feature_dim=args.cost_feature_dim,
            seed=args.seed + 12345,
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
        if official_fm is not None:
            print("official_otcfm=ExactOptimalTransportConditionalFlowMatcher(ot_method='exact')")
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
        coupled_metrics = {}
        if official_fm is not None:
            t, xt, ut = official_fm.sample_location_and_conditional_flow(x0, x1)
            coupled_metrics = {
                "mode": OFFICIAL_OTCFM_MODE,
                "context_size": int(local_batch),
                "source_context_size": int(local_batch),
                "target_context_size": int(local_batch),
                "feature_dim": int(x0.flatten(1).shape[1]),
                "official_otplan_sampler": True,
            }
        else:
            if ot_sampler is None:
                raise RuntimeError("ot_sampler is required unless official_fm is provided.")
            coupled = ot_sampler.sample_pairs(x0, x1, step=step)
            t = torch.rand(coupled.x0.shape[0], device=device)
            t_view = t.view(-1, 1, 1, 1)
            xt = (1.0 - t_view) * coupled.x0 + t_view * coupled.x1
            ut = coupled.x1 - coupled.x0
            coupled_metrics = coupled.metrics
        sync_if_cuda(device)
        ot_time = time.perf_counter() - ot_start

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

        should_validate = (
            args.val_every > 0
            and val_iter is not None
            and (official_fm is not None or val_ot_sampler is not None)
            and (step % args.val_every == 0 or step == args.total_steps - 1)
        )
        val_metrics = {}
        if should_validate:
            effective_val_batches = val_num_batches if args.val_batches <= 0 else args.val_batches
            sync_if_cuda(device)
            val_start = time.perf_counter()
            local_val_loss_sum, local_val_ot_sum, local_val_samples, local_val_batch_count = evaluate_validation_loss(
                net_model,
                val_iter,
                val_ot_sampler,
                device,
                val_batches=effective_val_batches,
                step=step,
                seed=args.seed,
                amp=args.amp,
                official_fm=official_fm,
            )
            sync_if_cuda(device)
            global_val_loss_sum = reduce_sum(local_val_loss_sum, device)
            global_val_ot_sum = reduce_sum(local_val_ot_sum, device)
            global_val_samples = reduce_sum(local_val_samples, device)
            global_val_batch_count = reduce_sum(local_val_batch_count, device)
            val_metrics = {
                "val_loss": global_val_loss_sum / max(global_val_samples, 1.0),
                "val_ot_time_s": global_val_ot_sum / max(global_val_batch_count, 1.0),
                "val_time_s": reduce_scalar(time.perf_counter() - val_start, device, op=dist.ReduceOp.MAX),
                "val_batches": int(effective_val_batches),
                "val_images": int(global_val_samples),
            }

        should_log = step % args.log_step == 0 or step == args.total_steps - 1 or should_validate
        if is_rank0 and should_log:
            images_per_s = args.batch_size / max(wall_step, 1e-12)
            postfix = {"loss": f"{mean_loss:.4g}", "ot_s": f"{mean_ot:.3f}", "img_s": f"{images_per_s:.1f}"}
            if val_metrics:
                postfix["val"] = f"{val_metrics['val_loss']:.4g}"
            progress.set_postfix(**postfix)
            row = {
                "step": int(step),
                "loss": mean_loss,
                "ot_time_s": mean_ot,
                "ot_s": mean_ot,
                "step_time_s": wall_step,
                "step_s": wall_step,
                "images_per_s": images_per_s,
                "images_s": images_per_s,
                "images/s": images_per_s,
                "peak_mem_gb": peak_memory_gb(device),
                "lr": float(sched.get_last_lr()[0]),
                **coupled_metrics,
                **val_metrics,
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
