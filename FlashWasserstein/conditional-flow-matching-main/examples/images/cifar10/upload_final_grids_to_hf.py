"""Upload final CIFAR-10 global OT-CFM sample grids to Hugging Face Hub.

This is a tiny convenience script for remote GPU nodes where opening PNGs is
awkward.  It finds the latest ``samples_step_*.png`` in each run directory and
uploads them to a dataset repo.

Example:

    HF_TOKEN=hf_... python examples/images/cifar10/upload_final_grids_to_hf.py \
      --run_root ~/FlashSinkhorn/output/cifar10_global_ot_50k \
      --repo_id USER/cifar10-global-ot-grids \
      --private \
      --endpoint https://hf-mirror.com

If ``hf-mirror.com`` does not support authenticated uploads from your node,
rerun without ``--endpoint`` or with ``--endpoint https://huggingface.co``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


RUN_ORDER = [
    "independent",
    "local_exact_pot",
    "local_entropic",
    "flash_global_entropic",
]


def sample_step(path: Path) -> int:
    match = re.search(r"samples_step_(\d+)\.png$", path.name)
    return int(match.group(1)) if match else -1


def run_rank(name: str) -> tuple[int, str]:
    for idx, prefix in enumerate(RUN_ORDER):
        if name.startswith(prefix):
            return idx, name
    return len(RUN_ORDER), name


def find_latest_grids(run_root: Path) -> list[Path]:
    grids = []
    for run_dir in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: run_rank(p.name)):
        candidates = sorted(run_dir.glob("samples_step_*.png"), key=sample_step)
        if candidates:
            grids.append(candidates[-1])
    return grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="Root containing the four CIFAR run dirs.")
    parser.add_argument("--repo_id", required=True, help="Dataset repo, e.g. USER/cifar10-global-ot-grids.")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--path_in_repo", default="cifar10_global_ot_50k")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    grids = find_latest_grids(run_root)
    if not grids:
        raise FileNotFoundError(f"No samples_step_*.png files found under {run_root}")
    if not args.token and not args.dry_run:
        raise ValueError("Set HF_TOKEN or pass --token.")

    manifest = {
        "run_root": str(run_root),
        "endpoint": args.endpoint,
        "repo_id": args.repo_id,
        "files": [],
    }
    for grid in grids:
        remote_name = f"{grid.parent.name}_{grid.name}"
        manifest["files"].append(
            {
                "local": str(grid),
                "path_in_repo": f"{args.path_in_repo}/{remote_name}",
                "step": sample_step(grid),
                "run": grid.parent.name,
            }
        )

    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return

    api = HfApi(endpoint=args.endpoint, token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    for item in manifest["files"]:
        print(f"Uploading {item['local']} -> {item['path_in_repo']}", flush=True)
        api.upload_file(
            path_or_fileobj=item["local"],
            path_in_repo=item["path_in_repo"],
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
        )

    manifest_path = run_root / "uploaded_sample_grids_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=f"{args.path_in_repo}/manifest.json",
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
    )
    print(f"Done: {args.endpoint}/{args.repo_id}/tree/{args.revision}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
