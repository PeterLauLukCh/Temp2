"""Build the shared latent projection statistics after parallel encoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encode_imagenet256_latents import calibrate_projection, save_reconstruction_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--projection_seed", type=int, default=0)
    parser.add_argument("--calibration_samples", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_reconstruction", action="store_true")
    parser.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--scaling_factor", type=float, default=0.18215)
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir).expanduser()
    shard_paths = sorted(latent_dir.glob("latents_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No latents_*.pt shards found in {latent_dir}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    start = time.perf_counter()
    projection_path = calibrate_projection(
        latent_dir,
        shard_paths,
        proj_dim=args.proj_dim,
        seed=args.projection_seed,
        max_samples=args.calibration_samples,
        device=device,
    )
    if args.save_reconstruction:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        save_reconstruction_grid(
            vae,
            shard_paths[0],
            latent_dir,
            device=device,
            scaling_factor=args.scaling_factor,
        )
    payload = {
        "latent_dir": str(latent_dir),
        "num_shards": len(shard_paths),
        "projection": projection_path.name,
        "proj_dim": args.proj_dim,
        "projection_seed": args.projection_seed,
        "calibration_samples": args.calibration_samples,
        "elapsed_s": time.perf_counter() - start,
    }
    (latent_dir / "projection_metadata.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
