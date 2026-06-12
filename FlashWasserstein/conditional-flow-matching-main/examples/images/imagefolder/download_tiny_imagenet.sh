#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-$HOME/datasets/tiny-imagenet}"
URL="https://cs231n.stanford.edu/tiny-imagenet-200.zip"

mkdir -p "$DEST"
cd "$DEST"

if [ ! -f tiny-imagenet-200.zip ]; then
  wget -O tiny-imagenet-200.zip "$URL"
fi

if [ ! -d tiny-imagenet-200 ]; then
  unzip -q tiny-imagenet-200.zip
fi

echo "Tiny ImageNet train ImageFolder root:"
echo "$DEST/tiny-imagenet-200/train"
