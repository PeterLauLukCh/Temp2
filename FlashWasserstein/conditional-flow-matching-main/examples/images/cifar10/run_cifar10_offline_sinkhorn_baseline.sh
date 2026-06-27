#!/usr/bin/env bash
set -euo pipefail

# Offline large-Sinkhorn-pair baseline for CIFAR-10.
#
# This is the paper-style offline baseline:
#   1. Precompute noise/data pairs with negative-dot-product Sinkhorn,
#      eps_tilde = std(C) * relative_eps.
#   2. Train a normal flow-matching model from the cached pairs, without online OT.
#   3. Optionally evaluate with CleanFID.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$CFM_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-/mindopt/ea120/datasets/cifar10}"
OUT_ROOT="${OUT_ROOT:-/mindopt/ea120/output/cifar10_offline_sinkhorn_dot_baseline}"
PAIR_ROOT="${PAIR_ROOT:-$OUT_ROOT/pairs}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_ROOT/train}"
EVAL_OUT="${EVAL_OUT:-$OUT_ROOT/eval}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/_logs}"

PRECOMPUTE="${PRECOMPUTE:-1}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
EVAL_GPU="${EVAL_GPU:-0}"

COUPLING_SIZE="${COUPLING_SIZE:-2048}"
RELATIVE_EPS="${RELATIVE_EPS:-0.01}"
NUM_PAIRS="${NUM_PAIRS:-1048576}"
MAX_ITERS="${MAX_ITERS:-50000}"
THRESHOLD="${THRESHOLD:-0.001}"
CHECK_EVERY="${CHECK_EVERY:-100}"
SAMPLE_CHUNK="${SAMPLE_CHUNK:-512}"
SHARD_PAIRS="${SHARD_PAIRS:-1048576}"
SEED="${SEED:-0}"

BATCH="${BATCH:-128}"
STEPS="${STEPS:-400001}"
PRIMARY_STEP="${PRIMARY_STEP:-400000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_STEP="${SAVE_STEP:-50000}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.0}"

EVAL_NFE_LIST="${EVAL_NFE_LIST:-100}"
EVAL_TRAIN_NUM_GEN="${EVAL_TRAIN_NUM_GEN:-50000}"
EVAL_TEST_NUM_GEN="${EVAL_TEST_NUM_GEN:-10000}"
EVAL_BATCH_SIZE_FID="${EVAL_BATCH_SIZE_FID:-1024}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$CFM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PAIR_DIR="${PAIR_DIR:-$PAIR_ROOT/cifar10_dot_n${COUPLING_SIZE}_eps${RELATIVE_EPS}_pairs${NUM_PAIRS}_seed${SEED}}"

count_gpus() {
  local group="$1"
  local spaced="${group//,/ }"
  # shellcheck disable=SC2086
  set -- $spaced
  echo "$#"
}

run_name() {
  printf "offline_sinkhorn_dot_n%s_eps%s_pairs%s_bs%s_seed%s" \
    "$COUPLING_SIZE" "$RELATIVE_EPS" "$NUM_PAIRS" "$BATCH" "$SEED"
}

main() {
  cd "$CFM_DIR"
  mkdir -p "$OUT_ROOT" "$PAIR_ROOT" "$TRAIN_OUT" "$EVAL_OUT" "$LOG_DIR"

  echo "=== CIFAR-10 offline Sinkhorn baseline ==="
  echo "cfm_dir=$CFM_DIR"
  echo "data_dir=$DATA_DIR"
  echo "pair_dir=$PAIR_DIR"
  echo "train_out=$TRAIN_OUT"
  echo "coupling_size=$COUPLING_SIZE relative_eps=$RELATIVE_EPS num_pairs=$NUM_PAIRS"
  echo "train_gpus=$TRAIN_GPUS batch=$BATCH steps=$STEPS"

  "$PYTHON_BIN" --version

  if [ "$PRECOMPUTE" = "1" ]; then
    if [ "$SKIP_DONE" = "1" ] && [ -f "$PAIR_DIR/metadata.json" ]; then
      echo "skip precompute: found $PAIR_DIR/metadata.json"
    else
      cmd=(
        "$PYTHON_BIN" examples/images/cifar10/precompute_cifar10_offline_sinkhorn_pairs.py
        --data_dir "$DATA_DIR"
        --out_dir "$PAIR_DIR"
        --num_pairs "$NUM_PAIRS"
        --coupling_size "$COUPLING_SIZE"
        --relative_eps "$RELATIVE_EPS"
        --max_iters "$MAX_ITERS"
        --threshold "$THRESHOLD"
        --check_every "$CHECK_EVERY"
        --sample_chunk "$SAMPLE_CHUNK"
        --shard_pairs "$SHARD_PAIRS"
        --seed "$SEED"
        --device cuda
      )
      printf 'precompute command: CUDA_VISIBLE_DEVICES=%q' "$PRECOMPUTE_GPU"
      printf ' %q' "${cmd[@]}"
      printf '\n'
      if [ "$DRY_RUN" != "1" ]; then
        CUDA_VISIBLE_DEVICES="$PRECOMPUTE_GPU" "${cmd[@]}" >"$LOG_DIR/precompute_n${COUPLING_SIZE}_eps${RELATIVE_EPS}.log" 2>&1
      fi
    fi
  fi

  if [ "$TRAIN" = "1" ]; then
    local nproc
    nproc="$(count_gpus "$TRAIN_GPUS")"
    local name
    name="$(run_name)"
    local ckpt="$TRAIN_OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
    if [ "$SKIP_DONE" = "1" ] && [ -f "$ckpt" ]; then
      echo "skip training: found $ckpt"
    else
      cmd=(
        torchrun --standalone --nproc_per_node="$nproc"
        examples/images/cifar10/train_cifar10_offline_pairs.py
        --data_dir "$DATA_DIR"
        --pair_dir "$PAIR_DIR"
        --output_dir "$TRAIN_OUT"
        --batch_size "$BATCH"
        --total_steps "$STEPS"
        --num_workers "$NUM_WORKERS"
        --amp
        --lr "$LR"
        --warmup "$WARMUP"
        --lr_schedule "$LR_SCHEDULE"
        --min_lr_ratio "$MIN_LR_RATIO"
        --save_step "$SAVE_STEP"
        --sample_every 0
        --log_step 20
        --seed "$SEED"
        --num_channel 128
        --num_res_blocks 2
        --channel_mult 1,2,2,2
        --attention_resolutions 16
        --num_heads 4
        --num_head_channels 64
        --dropout 0.1
      )
      printf 'train command: CUDA_VISIBLE_DEVICES=%q' "$TRAIN_GPUS"
      printf ' %q' "${cmd[@]}"
      printf '\n'
      if [ "$DRY_RUN" != "1" ]; then
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${cmd[@]}" >"$LOG_DIR/train_${name}.log" 2>&1
      fi
    fi
  fi

  if [ "$EVAL" = "1" ]; then
    local name
    name="$(run_name)"
    local ckpt="$TRAIN_OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
    mkdir -p "$EVAL_OUT/train50k" "$EVAL_OUT/test10k"
    for nfe in $EVAL_NFE_LIST; do
      if [ "$DRY_RUN" = "1" ]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" examples/images/cifar10/evaluate_cifar10_global_ot.py \
        --checkpoint "$ckpt" \
        --out_json "$EVAL_OUT/train50k/${name}_train50k_euler${nfe}.json" \
        --num_gen "$EVAL_TRAIN_NUM_GEN" \
        --batch_size_fid "$EVAL_BATCH_SIZE_FID" \
        --integration_method euler \
        --integration_steps "$nfe" \
        --dataset_split train \
        --fid_mode "$FID_MODE" \
        --device cuda \
        > "$LOG_DIR/eval_train50k_${name}_euler${nfe}.log" 2>&1

      CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" examples/images/cifar10/evaluate_cifar10_global_ot.py \
        --checkpoint "$ckpt" \
        --out_json "$EVAL_OUT/test10k/${name}_test10k_euler${nfe}.json" \
        --num_gen "$EVAL_TEST_NUM_GEN" \
        --batch_size_fid "$EVAL_BATCH_SIZE_FID" \
        --integration_method euler \
        --integration_steps "$nfe" \
        --dataset_split test \
        --fid_mode "$FID_MODE" \
        --device cuda \
        > "$LOG_DIR/eval_test10k_${name}_euler${nfe}.log" 2>&1
    done
  fi

  echo "done"
}

main "$@"

