#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_variance_4gpu_400k}"
LOG_DIR="${LOG_DIR:-$OUT/_logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-400001}"
PRIMARY_STEP="${PRIMARY_STEP:-400000}"
SEEDS="${SEEDS:-0 1 2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COST_DIM="${COST_DIM:-256}"
SAVE_STEP="${SAVE_STEP:-50000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-25000}"
LOG_STEP="${LOG_STEP:-20}"
VAL_EVERY="${VAL_EVERY:-5000}"
VAL_BATCHES="${VAL_BATCHES:-0}"
POT_NUM_THREADS="${POT_NUM_THREADS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL="${EVAL:-1}"
EVAL_OUT="${EVAL_OUT:-${OUT}_eval_fast}"
EVAL_NUM_GEN="${EVAL_NUM_GEN:-50000}"
EVAL_BATCH_SIZE_FID="${EVAL_BATCH_SIZE_FID:-1024}"
EVAL_NFE_LIST="${EVAL_NFE_LIST:-100}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-0 1 2 3}"
EVAL_COMPUTE_IS="${EVAL_COMPUTE_IS:-0}"
EVAL_IS_REQUIRED="${EVAL_IS_REQUIRED:-0}"
EVAL_IS_BATCH_SIZE="${EVAL_IS_BATCH_SIZE:-512}"
EVAL_IS_SPLITS="${EVAL_IS_SPLITS:-10}"
EVAL_SEED="${EVAL_SEED:-1234}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT_DIR"
mkdir -p "$DATA_DIR" "$OUT" "$LOG_DIR"

COMMON_BASE=(
  --data_dir "$DATA_DIR"
  --output_dir "$OUT"
  --batch_size "$BATCH"
  --total_steps "$STEPS"
  --num_workers "$NUM_WORKERS"
  --amp
  --cost_feature_dim "$COST_DIM"
  --lr 2e-4
  --warmup 5000
  --grad_clip 1.0
  --ema_decay 0.9999
  --save_step "$SAVE_STEP"
  --sample_every "$SAMPLE_EVERY"
  --sample_batch 64
  --integration_steps 100
  --log_step "$LOG_STEP"
  --val_every "$VAL_EVERY"
  --val_batches "$VAL_BATCHES"
  --num_channel 128
  --num_res_blocks 2
  --channel_mult 1,2,2,2
  --attention_resolutions 16
  --num_heads 4
  --num_head_channels 64
  --dropout 0.1
  --pot_num_threads "$POT_NUM_THREADS"
)

run_name() {
  local mode="$1"
  local context="$2"
  local eps="$3"
  local iters="$4"
  local seed="$5"
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" "$mode" "$context" "$eps" "$iters" "$BATCH" "$seed"
}

launch() {
  local seed="$1"
  local gpu="$2"
  local label="$3"
  local mode="$4"
  local context="$5"
  local eps="$6"
  local iters="$7"

  local name
  name="$(run_name "$mode" "$context" "$eps" "$iters" "$seed")"
  local done_ckpt="$OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
  local log="$LOG_DIR/seed${seed}_gpu${gpu}_${name}.log"
  local cmd=(
    "$PYTHON_BIN" examples/images/cifar10/train_cifar10_global_ot.py
    "${COMMON_BASE[@]}"
    --seed "$seed"
    --coupling_mode "$mode"
    --context_size "$context"
    --eps "$eps"
    --sinkhorn_iters "$iters"
  )

  echo "=== seed $seed GPU $gpu: $label ==="
  echo "run=$name"
  echo "log=$log"
  if [ "$SKIP_DONE" = "1" ] && [ -f "$done_ckpt" ]; then
    echo "skip: found $done_ckpt"
    return 0
  fi

  printf 'command: CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
  PIDS+=("$!")
  RUNS+=("seed${seed}:gpu${gpu}:${name}")
}

echo "=== CIFAR-10 4-GPU variance experiment ==="
echo "root=$ROOT_DIR"
echo "data_dir=$DATA_DIR"
echo "out=$OUT"
echo "logs=$LOG_DIR"
echo "seeds=$SEEDS"
echo "batch=$BATCH steps=$STEPS primary_step=$PRIMARY_STEP num_workers=$NUM_WORKERS"
echo "validation: every=$VAL_EVERY val_batches=$VAL_BATCHES (0 means full test set)"
echo "auto_eval=$EVAL eval_out=$EVAL_OUT eval_nfe=$EVAL_NFE_LIST"

for SEED in $SEEDS; do
  declare -a PIDS=()
  declare -a RUNS=()
  echo "=== starting training wave seed=$SEED ==="

  launch "$SEED" 0 "local exact POT ctx=128" local_exact_pot 128 0.05 20
  launch "$SEED" 1 "Flash ctx=8192 eps=0.03" flash_global_entropic 8192 0.03 30
  launch "$SEED" 2 "Flash ctx=16384 eps=0.03" flash_global_entropic 16384 0.03 30
  launch "$SEED" 3 "Flash ctx=32768 eps=0.03" flash_global_entropic 32768 0.03 30

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  status=0
  for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
      echo "done: ${RUNS[$idx]}"
    else
      code="$?"
      echo "failed: ${RUNS[$idx]} exit=$code"
      status=1
    fi
  done
  if [ "$status" != "0" ]; then
    echo "training failed in seed wave $SEED"
    exit "$status"
  fi

  if [ "$EVAL" = "1" ]; then
    echo "=== post-training eval seed=$SEED ==="
    INCLUDE_CSV="$(printf "local_exact_pot_ctx128_eps0.05_it20_bs%s_seed%s,flash_global_entropic_ctx8192_eps0.03_it30_bs%s_seed%s,flash_global_entropic_ctx16384_eps0.03_it30_bs%s_seed%s,flash_global_entropic_ctx32768_eps0.03_it30_bs%s_seed%s" "$BATCH" "$SEED" "$BATCH" "$SEED" "$BATCH" "$SEED" "$BATCH" "$SEED")"
    RUN_ROOT="$OUT" \
    OUT="$EVAL_OUT" \
    PYTHON_BIN="$PYTHON_BIN" \
    STEPS_LIST="$PRIMARY_STEP" \
    NFE_LIST="$EVAL_NFE_LIST" \
    NUM_GEN="$EVAL_NUM_GEN" \
    BATCH_SIZE_FID="$EVAL_BATCH_SIZE_FID" \
    GPU_LIST="$EVAL_GPU_LIST" \
    INCLUDE="$INCLUDE_CSV" \
    COMPUTE_IS="$EVAL_COMPUTE_IS" \
    IS_REQUIRED="$EVAL_IS_REQUIRED" \
    IS_BATCH_SIZE="$EVAL_IS_BATCH_SIZE" \
    IS_SPLITS="$EVAL_IS_SPLITS" \
    SEED="$EVAL_SEED" \
      ./examples/images/cifar10/run_cifar10_fast_metric_eval.sh
  fi
done

echo "done: CIFAR-10 variance experiment"
