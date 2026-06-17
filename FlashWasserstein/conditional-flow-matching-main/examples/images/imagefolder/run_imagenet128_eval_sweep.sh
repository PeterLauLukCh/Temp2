#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/imagenet-1k-256x256}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/imagenet128_formal_250k}"
OUT_DIR="${OUT_DIR:-$HOME/FlashSinkhorn/output/imagenet128_formal_250k_eval}"
NPROC="${NPROC:-8}"
EVAL_BATCH="${EVAL_BATCH:-256}"
STEPS_LIST="${STEPS_LIST:-100000 200000 250000}"
NFE_LIST="${NFE_LIST:-25 50 100}"

if ! compgen -G "$DATA_DIR/validation-*.parquet" >/dev/null; then
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python examples/images/imagefolder/download_hf_imagenet256.py \
    --dest "$DATASET_ROOT" \
    --split validation
fi

for STEP in $STEPS_LIST; do
  for NFE in $NFE_LIST; do
    echo "=== ImageNet-128 eval step=$STEP Euler NFE=$NFE ==="
    torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/evaluate_hf_parquet_folders.py \
      --run_root "$RUN_ROOT" \
      --data_dir "$DATA_DIR" \
      --out_dir "$OUT_DIR" \
      --step "$STEP" \
      --image_size 128 \
      --num_gen 50000 \
      --num_real 50000 \
      --batch_size "$EVAL_BATCH" \
      --integration_method euler \
      --integration_steps "$NFE" \
      --reference_split validation \
      --fallback_reference_split train \
      --reference_mode balanced \
      --label_mode balanced \
      --compute_kid \
      --amp \
      --amp_dtype bf16
  done
done

echo "done: $OUT_DIR"
