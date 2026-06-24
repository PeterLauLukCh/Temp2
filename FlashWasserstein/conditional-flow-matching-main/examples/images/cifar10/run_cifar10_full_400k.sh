#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_full_400k}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-400001}"
BATCH="${BATCH:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-256}"
FLASH_CONTEXT="${FLASH_CONTEXT:-32768}"
METHODS="${METHODS:-independent local_exact_pot local_entropic flash_global_entropic}"
SAVE_STEP="${SAVE_STEP:-50000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-25000}"
LOG_STEP="${LOG_STEP:-20}"
VAL_EVERY="${VAL_EVERY:-5000}"
VAL_BATCHES="${VAL_BATCHES:-0}"
POT_NUM_THREADS="${POT_NUM_THREADS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT_DIR"
mkdir -p "$DATA_DIR" "$OUT"

# With NPROC=8, BATCH=1024 gives local batch 128, matching the TorchCFM
# CIFAR-10 recipe while using all GPUs. Override BATCH only for a large-batch
# scaling ablation.
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
)

run_name() {
  local mode="$1"
  local context="$2"
  local eps="$3"
  local iters="$4"
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" "$mode" "$context" "$eps" "$iters" "$BATCH" "$SEED"
}

run_train() {
  local label="$1"
  local mode="$2"
  local context="$3"
  local eps="$4"
  local iters="$5"
  shift 5

  local name
  name="$(run_name "$mode" "$context" "$eps" "$iters")"
  local eval_ckpt="$OUT/$name/weights_step_$(printf "%08d" "$((STEPS - 1))").pt"
  local final_ckpt="$OUT/$name/weights_step_$(printf "%08d" "$STEPS").pt"

  echo "=== CIFAR-10 $label ==="
  echo "run=$name"
  if [ "$SKIP_DONE" = "1" ] && { [ -f "$eval_ckpt" ] || [ -f "$final_ckpt" ]; }; then
    echo "skip: found completed checkpoint for $name"
    return 0
  fi

  local cmd=(
    torchrun --standalone --nproc_per_node="$NPROC"
    examples/images/cifar10/train_cifar10_global_ot.py
    "${COMMON[@]}"
    --coupling_mode "$mode"
    --context_size "$context"
    --eps "$eps"
    --sinkhorn_iters "$iters"
    "$@"
  )

  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "${cmd[@]}"
}

echo "=== CIFAR-10 full 400k launcher ==="
echo "root=$ROOT_DIR"
echo "data_dir=$DATA_DIR"
echo "out=$OUT"
echo "methods=$METHODS"
echo "nproc=$NPROC batch=$BATCH steps=$STEPS cost_dim=$COST_DIM flash_context=$FLASH_CONTEXT"
echo "validation: every=$VAL_EVERY batches=$VAL_BATCHES"

for METHOD in $METHODS; do
  case "$METHOD" in
    independent)
      run_train "independent FM" independent 8192 0.05 20
      ;;
    local_exact_pot|ot_cfm|pot)
      run_train "local exact POT OT-CFM" local_exact_pot 8192 0.05 20 --pot_num_threads "$POT_NUM_THREADS"
      ;;
    local_entropic)
      run_train "local entropic OT-CFM" local_entropic 8192 0.01 30
      ;;
    flash_global_entropic|flash|flash32768)
      run_train "Flash global entropic OT-CFM" flash_global_entropic "$FLASH_CONTEXT" 0.01 30
      ;;
    *)
      echo "Unknown method '$METHOD'. Use independent, local_exact_pot, local_entropic, or flash_global_entropic." >&2
      exit 2
      ;;
  esac
done

echo "done: $OUT"
