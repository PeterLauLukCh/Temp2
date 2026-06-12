"""Download Hugging Face ImageNet-1K 256x256 Parquet shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path


REPO_ID = "benjamin-paine/imagenet-1k-256x256"


def endpoint_base() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def api_url() -> str:
    return f"{endpoint_base()}/api/datasets/{REPO_ID}/tree/main?recursive=1"


def resolve_url() -> str:
    return f"{endpoint_base()}/datasets/{REPO_ID}/resolve/main"


def list_files(split: str):
    with urllib.request.urlopen(api_url(), timeout=60) as response:
        entries = json.load(response)
    prefix = f"data/{split}-"
    files = [
        entry
        for entry in entries
        if entry.get("path", "").startswith(prefix)
        and entry.get("path", "").endswith(".parquet")
    ]
    return sorted(files, key=lambda entry: entry["path"])


def download_file(url: str, dest: Path, expected_size: int | None) -> None:
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"skip {dest.name}", flush=True)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        start = time.perf_counter()
        with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        size = tmp_path.stat().st_size
        if expected_size is not None and size != expected_size:
            raise IOError(f"{dest.name}: got {size} bytes, expected {expected_size}")
        tmp_path.replace(dest)
        elapsed = time.perf_counter() - start
        print(f"downloaded {dest.name} {size / 1e9:.3f} GB in {elapsed:.1f}s", flush=True)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="~/datasets/imagenet-1k-256x256")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    files = list_files(args.split)
    if not files:
        raise RuntimeError(f"No {args.split} Parquet shards found for {REPO_ID}")

    total = sum(entry.get("size") or 0 for entry in files)
    print(f"{REPO_ID} split={args.split} shards={len(files)} bytes={total}", flush=True)
    for entry in files:
        path = entry["path"]
        url = f"{resolve_url()}/{path}"
        download_file(url, dest / path, entry.get("size"))

    print(f"done: {dest / 'data'}", flush=True)


if __name__ == "__main__":
    main()
