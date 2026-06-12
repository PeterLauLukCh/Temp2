#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-$HOME/datasets/imagenet}"
COMPETITION="imagenet-object-localization-challenge"
RAW_DIR="$DEST/raw"
EXTRACT_DIR="$DEST/extracted"
TRAIN_OUT="$DEST/train"

echo "Destination: $DEST"
mkdir -p "$RAW_DIR" "$EXTRACT_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI is not on PATH. Install it with: python -m pip install kaggle" >&2
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: missing $HOME/.kaggle/kaggle.json" >&2
  echo "Create a Kaggle API token, put it there, then run: chmod 600 $HOME/.kaggle/kaggle.json" >&2
  exit 1
fi

echo "Downloading ImageNet-1K / ILSVRC object-localization archive from Kaggle..."
kaggle competitions download -c "$COMPETITION" -p "$RAW_DIR"

ZIP_PATH="$RAW_DIR/${COMPETITION}.zip"
if [ -f "$ZIP_PATH" ]; then
  echo "Unzipping Kaggle wrapper archive..."
  python -m zipfile -e "$ZIP_PATH" "$RAW_DIR/kaggle"
else
  echo "No wrapper zip found at $ZIP_PATH; continuing with existing files under $RAW_DIR"
fi

find_train_root() {
  python - "$RAW_DIR" "$EXTRACT_DIR" <<'PY'
import os
import sys

roots = sys.argv[1:]
for root in roots:
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if base == "train":
            wnids = [d for d in dirnames if d.startswith("n") and len(d) == 9]
            if len(wnids) >= 100:
                print(dirpath)
                raise SystemExit(0)
raise SystemExit(1)
PY
}

if TRAIN_ROOT="$(find_train_root 2>/dev/null)"; then
  echo "Found ImageFolder train root: $TRAIN_ROOT"
  echo "Use this with --data_root:"
  echo "$TRAIN_ROOT"
  exit 0
fi

PATCHED_TAR="$(find "$RAW_DIR" -name 'imagenet_object_localization_patched2019.tar.gz' -print -quit)"
if [ -n "$PATCHED_TAR" ]; then
  echo "Extracting patched Kaggle tarball. This can take a while..."
  tar -xzf "$PATCHED_TAR" -C "$EXTRACT_DIR"
  if TRAIN_ROOT="$(find_train_root 2>/dev/null)"; then
    echo "Found ImageFolder train root: $TRAIN_ROOT"
    echo "Use this with --data_root:"
    echo "$TRAIN_ROOT"
    exit 0
  fi
fi

TRAIN_TAR="$(find "$RAW_DIR" "$EXTRACT_DIR" -name 'ILSVRC2012_img_train.tar' -print -quit)"
if [ -z "$TRAIN_TAR" ]; then
  echo "ERROR: could not find an ImageFolder train tree or ILSVRC2012_img_train.tar." >&2
  echo "Inspect $RAW_DIR and $EXTRACT_DIR; the Kaggle archive format may have changed." >&2
  exit 1
fi

echo "Extracting nested ILSVRC2012 train archive..."
mkdir -p "$EXTRACT_DIR/train_tars" "$TRAIN_OUT"
tar -xf "$TRAIN_TAR" -C "$EXTRACT_DIR/train_tars"

echo "Expanding per-class tar files into $TRAIN_OUT ..."
for class_tar in "$EXTRACT_DIR"/train_tars/n*.tar; do
  [ -f "$class_tar" ] || continue
  wnid="$(basename "$class_tar" .tar)"
  mkdir -p "$TRAIN_OUT/$wnid"
  tar -xf "$class_tar" -C "$TRAIN_OUT/$wnid"
done

echo "Done."
echo "Use this with --data_root:"
echo "$TRAIN_OUT"
