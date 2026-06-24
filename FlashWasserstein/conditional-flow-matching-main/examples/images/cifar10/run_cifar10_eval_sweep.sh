#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/cifar10_full_400k}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_full_400k_eval}"
DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
STEPS_LIST="${STEPS_LIST:-100000 200000 400000}"
NFE_LIST="${NFE_LIST:-10 20 50 100}"
NUM_GEN="${NUM_GEN:-50000}"
NPROC="${NPROC:-8}"
BATCH_PER_GPU="${BATCH_PER_GPU:-${BATCH:-1024}}"
INCLUDE="${INCLUDE:-}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"
SCORE_REFERENCE="${SCORE_REFERENCE:-cleanfid_stats}"
KID_REFERENCE="${KID_REFERENCE:-folder}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
DATASET_RES="${DATASET_RES:-32}"

INCLUDE_FLAG=""
if [ -n "$INCLUDE" ]; then
  INCLUDE_FLAG="--include $INCLUDE"
fi
AMP_FLAG=""
if [ "$AMP" = "1" ]; then
  AMP_FLAG="--amp --amp_dtype $AMP_DTYPE"
fi

for STEP in $STEPS_LIST; do
  for NFE in $NFE_LIST; do
    echo "=== CIFAR-10 eval step=$STEP Euler NFE=$NFE ==="
    torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/evaluate_cifar10_folders.py \
      --run_root "$RUN_ROOT" \
      --step "$STEP" \
      --out_dir "$OUT" \
      --data_dir "$DATA_DIR" \
      --split train \
      --num_gen "$NUM_GEN" \
      --batch_size "$BATCH_PER_GPU" \
      --integration_method euler \
      --integration_steps "$NFE" \
      --score_reference "$SCORE_REFERENCE" \
      --kid_reference "$KID_REFERENCE" \
      --fid_mode "$FID_MODE" \
      --dataset_name "$DATASET_NAME" \
      --dataset_res "$DATASET_RES" \
      --compute_kid \
      $AMP_FLAG \
      $INCLUDE_FLAG
  done
done

echo "done: $OUT"
