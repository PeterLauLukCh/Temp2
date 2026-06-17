#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/imagenet-1k-256x256}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/imagenet64_full_250k}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/imagenet64_full_250k_eval}"
NPROC="${NPROC:-8}"
STEPS_LIST="${STEPS_LIST:-100000 200000 250000}"
NFE_LIST="${NFE_LIST:-25 50 100}"
NUM_GEN="${NUM_GEN:-50000}"
NUM_REAL="${NUM_REAL:-50000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"
INCLUDE="${INCLUDE:-}"

INCLUDE_FLAG=""
if [ -n "$INCLUDE" ]; then
  INCLUDE_FLAG="--include $INCLUDE"
fi

for STEP in $STEPS_LIST; do
  for NFE in $NFE_LIST; do
    echo "=== ImageNet-64 eval step=$STEP Euler NFE=$NFE ==="
    torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/evaluate_hf_parquet_folders.py \
      --run_root "$RUN_ROOT" \
      --data_dir "$DATA_DIR" \
      --out_dir "$OUT" \
      --step "$STEP" \
      --image_size 64 \
      --num_gen "$NUM_GEN" \
      --num_real "$NUM_REAL" \
      --batch_size "$BATCH_PER_GPU" \
      --integration_method euler \
      --integration_steps "$NFE" \
      --reference_split validation \
      --fallback_reference_split train \
      --reference_mode balanced \
      --label_mode balanced \
      --amp \
      --amp_dtype bf16 \
      --compute_kid \
      $INCLUDE_FLAG
  done
done

echo "done: $OUT"
