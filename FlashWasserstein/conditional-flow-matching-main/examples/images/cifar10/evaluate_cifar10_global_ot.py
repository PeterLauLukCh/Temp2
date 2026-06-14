"""Evaluate CIFAR-10 global OT-CFM checkpoints with clean-FID.

The original CIFAR evaluator in this repo assumes the historical checkpoint
layout.  This script accepts either one direct checkpoint or a run root
containing the standardized global-OT experiment directories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from cleanfid import fid
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402
from torchcfm.ot_coupling import parse_csv_ints  # noqa: E402


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    if "weights_step_" not in stem:
        return -1
    return int(stem.rsplit("weights_step_", 1)[1])


def find_checkpoints(run_root: Path, step: int | None) -> list[Path]:
    checkpoints = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if step is None:
            candidates = sorted(run_dir.glob("weights_step_*.pt"), key=checkpoint_step)
            if candidates:
                checkpoints.append(candidates[-1])
        else:
            path = run_dir / f"weights_step_{step:08d}.pt"
            if path.exists():
                checkpoints.append(path)
    return checkpoints


def build_model(train_args) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=getattr(train_args, "num_res_blocks", 2),
        num_channels=getattr(train_args, "num_channel", 128),
        channel_mult=parse_csv_ints(getattr(train_args, "channel_mult", "1,2,2,2")),
        num_heads=getattr(train_args, "num_heads", 4),
        num_head_channels=getattr(train_args, "num_head_channels", 64),
        attention_resolutions=getattr(train_args, "attention_resolutions", "16"),
        dropout=getattr(train_args, "dropout", 0.1),
    )


@torch.no_grad()
def euler_integrate(model, x: torch.Tensor, steps: int) -> torch.Tensor:
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t = torch.full((x.shape[0],), idx / float(steps), device=x.device)
        x = x + dt * model(t, x)
    return x


@torch.no_grad()
def heun_integrate(model, x: torch.Tensor, steps: int) -> torch.Tensor:
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t0 = torch.full((x.shape[0],), idx / float(steps), device=x.device)
        t1 = torch.full((x.shape[0],), (idx + 1) / float(steps), device=x.device)
        v0 = model(t0, x)
        x_pred = x + dt * v0
        v1 = model(t1, x_pred)
        x = x + 0.5 * dt * (v0 + v1)
    return x


@torch.no_grad()
def dopri5_integrate(model, x: torch.Tensor, tol: float) -> torch.Tensor:
    from torchdiffeq import odeint

    t_span = torch.tensor([0.0, 1.0], device=x.device)
    traj = odeint(model, x, t_span, rtol=tol, atol=tol, method="dopri5")
    return traj[-1]


def make_generator(model, args, device: torch.device):
    def gen(_unused_latent):
        with torch.no_grad():
            x = torch.randn(args.batch_size_fid, 3, 32, 32, device=device)
            if args.integration_method == "euler":
                x = euler_integrate(model, x, args.integration_steps)
            elif args.integration_method == "heun":
                x = heun_integrate(model, x, args.integration_steps)
            elif args.integration_method == "dopri5":
                x = dopri5_integrate(model, x, args.tol)
            else:
                raise ValueError(f"unknown integration method {args.integration_method}")
            return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)

    return gen


def save_preview(model, path: Path, args, device: torch.device) -> None:
    with torch.no_grad():
        x = torch.randn(args.preview_batch, 3, 32, 32, device=device)
        if args.integration_method == "euler":
            x = euler_integrate(model, x, args.integration_steps)
        elif args.integration_method == "heun":
            x = heun_integrate(model, x, args.integration_steps)
        elif args.integration_method == "dopri5":
            x = dopri5_integrate(model, x, args.tol)
        image = x.float().clamp(-1, 1).add(1).mul(0.5).cpu()
    save_image(image, path, nrow=int(math.sqrt(args.preview_batch)))


def evaluate_checkpoint(path: Path, args, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    train_args = argparse.Namespace(**checkpoint.get("args", {}))
    model = build_model(train_args).to(device)
    state = checkpoint.get("ema_model", checkpoint.get("net_model"))
    model.load_state_dict(strip_module_prefix(state))
    model.eval()

    if args.preview_dir:
        preview_dir = Path(args.preview_dir).expanduser()
        preview_dir.mkdir(parents=True, exist_ok=True)
        save_preview(model, preview_dir / f"{path.parent.name}.png", args, device)

    start = time.perf_counter()
    score = fid.compute_fid(
        gen=make_generator(model, args, device),
        dataset_name="cifar10",
        batch_size=args.batch_size_fid,
        dataset_res=32,
        num_gen=args.num_gen,
        dataset_split=args.dataset_split,
        mode=args.fid_mode,
    )
    elapsed = time.perf_counter() - start
    result = {
        "run": path.parent.name,
        "checkpoint": str(path),
        "step": checkpoint.get("step", checkpoint_step(path)),
        "fid": float(score),
        "num_gen": int(args.num_gen),
        "integration_method": args.integration_method,
        "integration_steps": int(args.integration_steps),
        "elapsed_s": elapsed,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--run_root")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--preview_dir", default="")
    parser.add_argument("--preview_batch", type=int, default=64)
    parser.add_argument("--num_gen", type=int, default=50000)
    parser.add_argument("--batch_size_fid", type=int, default=1024)
    parser.add_argument("--integration_method", choices=["euler", "heun", "dopri5"], default="euler")
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--fid_mode", default="legacy_tensorflow")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint).expanduser()]
    else:
        checkpoints = find_checkpoints(Path(args.run_root).expanduser(), args.step)
    if not checkpoints:
        raise FileNotFoundError("No matching checkpoints found.")

    results = []
    for path in checkpoints:
        print(f"Evaluating {path}", flush=True)
        result = evaluate_checkpoint(path, args, device)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
