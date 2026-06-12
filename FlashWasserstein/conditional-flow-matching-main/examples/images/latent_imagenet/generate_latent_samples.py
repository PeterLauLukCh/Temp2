"""Generate decoded images from a latent ImageNet flow checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_ot import parse_csv_ints  # noqa: E402
from torchcfm.models.unet.unet import UNetModelWrapper  # noqa: E402


def build_model(args) -> UNetModelWrapper:
    return UNetModelWrapper(
        dim=(4, 32, 32),
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channel,
        channel_mult=parse_csv_ints(args.channel_mult),
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
    )


@torch.no_grad()
def sample_latents(model, *, batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
    z = torch.randn(batch_size, 4, 32, 32, device=device)
    dt = 1.0 / float(steps)
    for idx in range(steps):
        t = torch.full((batch_size,), idx / float(steps), device=device)
        z = z + dt * model(t, z)
    return z


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--integration_steps", type=int, default=100)
    parser.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--scaling_factor", type=float, default=0.18215)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid_every", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    train_args = argparse.Namespace(**checkpoint.get("args", {}))
    for key, default in {
        "num_channel": 128,
        "num_res_blocks": 2,
        "channel_mult": "1,2,2,2",
        "num_heads": 4,
        "num_head_channels": 64,
        "attention_resolutions": "16",
        "dropout": 0.1,
    }.items():
        if not hasattr(train_args, key):
            setattr(train_args, key, default)

    model = build_model(train_args).to(device)
    state = checkpoint.get("ema_model", checkpoint.get("net_model"))
    model.load_state_dict(strip_module_prefix(state))
    model.eval()

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    out_dir = Path(args.out_dir).expanduser()
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    produced = 0
    grid_images = []
    while produced < args.num_samples:
        batch = min(args.batch_size, args.num_samples - produced)
        z = sample_latents(model, batch_size=batch, steps=args.integration_steps, device=device)
        decoded = vae.decode(z / float(args.scaling_factor)).sample
        images = decoded.float().clamp(-1, 1).add(1).mul(0.5).cpu()
        for idx, image in enumerate(images):
            save_image(image, image_dir / f"{produced + idx:08d}.png")
        if args.grid_every > 0 and produced % args.grid_every == 0:
            grid_images.append(images[: min(16, images.shape[0])])
        produced += batch
        print(f"generated {produced}/{args.num_samples}", flush=True)

    if grid_images:
        grid = torch.cat(grid_images, dim=0)[:64]
        save_image(grid, out_dir / "sample_grid.png", nrow=int(math.sqrt(grid.shape[0])))


if __name__ == "__main__":
    main()
