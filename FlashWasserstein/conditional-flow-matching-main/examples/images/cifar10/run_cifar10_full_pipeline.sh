#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/cifar10_full_400k}"
EVAL_OUT="${EVAL_OUT:-$HOME/FlashSinkhorn/output/cifar10_full_400k_eval}"
NPROC="${NPROC:-8}"
METHODS="${METHODS:-independent local_exact_pot local_entropic flash_global_entropic}"
STEPS="${STEPS:-400001}"
BATCH="${BATCH:-1024}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-256}"
FLASH_CONTEXT="${FLASH_CONTEXT:-32768}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
EVAL_STEPS_LIST="${EVAL_STEPS_LIST:-100000 200000 400000}"
NFE_LIST="${NFE_LIST:-10 20 50 100}"
NUM_GEN="${NUM_GEN:-50000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1024}"

echo "=== CIFAR-10 full pipeline ==="
echo "methods=$METHODS"
echo "train_steps=$STEPS train_batch=$BATCH nproc=$NPROC"
echo "run_root=$RUN_ROOT"
echo "eval_out=$EVAL_OUT"

if [ "$TRAIN" = "1" ]; then
  DATA_DIR="$DATA_DIR" \
  OUT="$RUN_ROOT" \
  NPROC="$NPROC" \
  METHODS="$METHODS" \
  STEPS="$STEPS" \
  BATCH="$BATCH" \
  SEED="$SEED" \
  COST_DIM="$COST_DIM" \
  FLASH_CONTEXT="$FLASH_CONTEXT" \
    ./examples/images/cifar10/run_cifar10_full_400k.sh
fi

if [ "$EVAL" = "1" ]; then
  RUN_ROOT="$RUN_ROOT" \
  OUT="$EVAL_OUT" \
  DATA_DIR="$DATA_DIR" \
  NPROC="$NPROC" \
  STEPS_LIST="$EVAL_STEPS_LIST" \
  NFE_LIST="$NFE_LIST" \
  NUM_GEN="$NUM_GEN" \
  BATCH_PER_GPU="$BATCH_PER_GPU" \
  INCLUDE="$(echo "$METHODS" | tr ' ' ',')" \
    ./examples/images/cifar10/run_cifar10_eval_sweep.sh
fi

echo "done: CIFAR-10 full pipeline"
