#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/cifar10_full_400k}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_full_400k_eval}"
DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
STEPS_LIST="${STEPS_LIST:-100000 200000 400000}"
NFE_LIST="${NFE_LIST:-25 50 100}"
NUM_GEN="${NUM_GEN:-50000}"
BATCH="${BATCH:-2048}"
INCLUDE="${INCLUDE:-}"

INCLUDE_FLAG=""
if [ -n "$INCLUDE" ]; then
  INCLUDE_FLAG="--include $INCLUDE"
fi

for STEP in $STEPS_LIST; do
  for NFE in $NFE_LIST; do
    echo "=== CIFAR-10 eval step=$STEP Euler NFE=$NFE ==="
    python examples/images/cifar10/evaluate_cifar10_folders.py \
      --run_root "$RUN_ROOT" \
      --step "$STEP" \
      --out_dir "$OUT" \
      --data_dir "$DATA_DIR" \
      --split train \
      --num_gen "$NUM_GEN" \
      --batch_size "$BATCH" \
      --integration_method euler \
      --integration_steps "$NFE" \
      --compute_kid \
      $INCLUDE_FLAG
  done
done

echo "done: $OUT"
