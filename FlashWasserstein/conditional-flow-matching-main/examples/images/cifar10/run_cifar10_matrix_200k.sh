#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_matrix_200k}"
LOG_DIR="${LOG_DIR:-$OUT/_logs}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-200001}"
PRIMARY_STEP="${PRIMARY_STEP:-200000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-256}"
SAVE_STEP="${SAVE_STEP:-50000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-25000}"
LOG_STEP="${LOG_STEP:-20}"
VAL_EVERY="${VAL_EVERY:-5000}"
VAL_BATCHES="${VAL_BATCHES:-0}"
POT_NUM_THREADS="${POT_NUM_THREADS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
WAIT="${WAIT:-1}"
EVAL="${EVAL:-1}"
EVAL_OUT="${EVAL_OUT:-${OUT}_eval_paper}"
EVAL_NUM_GEN="${EVAL_NUM_GEN:-50000}"
EVAL_NPROC="${EVAL_NPROC:-8}"
EVAL_BATCH_PER_GPU="${EVAL_BATCH_PER_GPU:-512}"
EVAL_NFE_LIST="${EVAL_NFE_LIST:-10 20 50 100}"
EVAL_STEPS_LIST="${EVAL_STEPS_LIST:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT_DIR"
mkdir -p "$DATA_DIR" "$OUT" "$LOG_DIR"

COMMON=(
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
  --seed "$SEED"
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
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" "$mode" "$context" "$eps" "$iters" "$BATCH" "$SEED"
}

declare -a PIDS=()
declare -a RUNS=()

launch() {
  local gpu="$1"
  local label="$2"
  local mode="$3"
  local context="$4"
  local eps="$5"
  local iters="$6"

  local name
  name="$(run_name "$mode" "$context" "$eps" "$iters")"
  local done_ckpt="$OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
  local log="$LOG_DIR/gpu${gpu}_${name}.log"
  local cmd=(
    python examples/images/cifar10/train_cifar10_global_ot.py
    "${COMMON[@]}"
    --coupling_mode "$mode"
    --context_size "$context"
    --eps "$eps"
    --sinkhorn_iters "$iters"
  )

  echo "=== GPU $gpu: $label ==="
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
  RUNS+=("gpu${gpu}:${name}")
}

echo "=== CIFAR-10 single-GPU matrix ==="
echo "root=$ROOT_DIR"
echo "data_dir=$DATA_DIR"
echo "out=$OUT"
echo "logs=$LOG_DIR"
echo "batch=$BATCH steps=$STEPS primary_step=$PRIMARY_STEP seed=$SEED"
echo "validation: every=$VAL_EVERY val_batches=$VAL_BATCHES (0 means full test set)"
echo "auto_eval=$EVAL eval_out=$EVAL_OUT"

launch 0 "independent FM baseline" independent 128 0.05 20
launch 1 "local exact POT OT-CFM" local_exact_pot 128 0.05 20
launch 2 "local entropic OT-CFM eps=0.01" local_entropic 128 0.01 30
launch 3 "Flash global entropic ctx=16384 eps=0.01" flash_global_entropic 16384 0.01 30
launch 4 "Flash global entropic ctx=32768 eps=0.01" flash_global_entropic 32768 0.01 30
launch 5 "Flash global entropic ctx=65536 eps=0.01" flash_global_entropic 65536 0.01 30
launch 6 "Flash global entropic ctx=32768 eps=0.02" flash_global_entropic 32768 0.02 30
launch 7 "Flash global entropic ctx=32768 eps=0.005" flash_global_entropic 32768 0.005 30

if [ "$DRY_RUN" = "1" ]; then
  echo "dry run complete"
  exit 0
fi

echo "started ${#PIDS[@]} jobs"
for idx in "${!PIDS[@]}"; do
  echo "${RUNS[$idx]} pid=${PIDS[$idx]}"
done

if [ "$WAIT" != "1" ]; then
  echo "WAIT=$WAIT, leaving jobs in background"
  exit 0
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
  echo "training failed; skip evaluation"
  exit "$status"
fi

if [ "$EVAL" = "1" ]; then
  if [ -z "$EVAL_STEPS_LIST" ]; then
    if [ "$PRIMARY_STEP" -ge 400000 ]; then
      EVAL_STEPS_LIST="100000 200000 400000"
    elif [ "$PRIMARY_STEP" -ge 200000 ]; then
      EVAL_STEPS_LIST="100000 200000"
    else
      EVAL_STEPS_LIST="$PRIMARY_STEP"
    fi
  fi

  echo "=== CIFAR-10 matrix post-training eval ==="
  RUN_ROOT="$OUT" \
  OUT="$EVAL_OUT" \
  DATA_DIR="$DATA_DIR" \
  STEPS_LIST="$EVAL_STEPS_LIST" \
  NFE_LIST="$EVAL_NFE_LIST" \
  NUM_GEN="$EVAL_NUM_GEN" \
  NPROC="$EVAL_NPROC" \
  BATCH_PER_GPU="$EVAL_BATCH_PER_GPU" \
    ./examples/images/cifar10/run_cifar10_matrix_eval.sh
fi

echo "done: CIFAR-10 matrix training"
