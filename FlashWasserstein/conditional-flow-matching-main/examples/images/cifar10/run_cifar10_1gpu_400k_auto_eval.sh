#!/usr/bin/env bash
set -euo pipefail

# CIFAR-10 one-GPU-per-run launcher with automatic FID evaluation.
#
# Matrix:
#   GPU 0: official OT-CFM exact, batch 128, 400k steps
#   GPU 1: FlashSinkhorn 1K eps=0.03, full-pixel cost, batch 128, 400k steps
#   GPU 2: FlashSinkhorn 2K eps=0.03, full-pixel cost, batch 128, 400k steps
#
# After all training jobs finish:
#   1. Evaluate 50k generated samples against CIFAR-10 train CleanFID stats.
#   2. Evaluate 10k generated samples against CIFAR-10 test CleanFID stats.
#
# Examples:
#   nohup ./examples/images/cifar10/run_cifar10_1gpu_400k_auto_eval.sh \
#     > /mindopt/ea120/output/cifar10_1gpu_bs128_exact_flash1k2k_400k.master.log 2>&1 &
#
#   TRAIN=0 EVAL=1 ./examples/images/cifar10/run_cifar10_1gpu_400k_auto_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$CFM_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-/mindopt/ea120/datasets/cifar10}"
OUT="${OUT:-/mindopt/ea120/output/cifar10_1gpu_bs128_exact_flash1k2k_400k}"
LOG_DIR="${LOG_DIR:-$OUT/_logs}"
EVAL_OUT="${EVAL_OUT:-${OUT}_eval_auto}"

TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
PREPARE_CIFAR="${PREPARE_CIFAR:-1}"
DOWNLOAD_CLEANFID_STATS="${DOWNLOAD_CLEANFID_STATS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"

BATCH="${BATCH:-128}"
STEPS="${STEPS:-400001}"
PRIMARY_STEP="${PRIMARY_STEP:-400000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
SAVE_STEP="${SAVE_STEP:-50000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"
LOG_STEP="${LOG_STEP:-20}"
VAL_EVERY="${VAL_EVERY:-0}"
POT_NUM_THREADS="${POT_NUM_THREADS:-1}"

LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.0}"

TRAIN_EVAL_NFE="${TRAIN_EVAL_NFE:-100}"
TEST_EVAL_NFE="${TEST_EVAL_NFE:-100}"
TRAIN_EVAL_NUM_GEN="${TRAIN_EVAL_NUM_GEN:-50000}"
TEST_EVAL_NUM_GEN="${TEST_EVAL_NUM_GEN:-10000}"
EVAL_BATCH_SIZE_FID="${EVAL_BATCH_SIZE_FID:-1024}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"
EVAL_SEED="${EVAL_SEED:-1234}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$CFM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

GPUS=(0 1 2)
MODES=(official_otcfm_exact flash_global_entropic flash_global_entropic)
CONTEXTS=(128 1024 2048)
EPS_VALUES=(0.05 0.03 0.03)
ITERS=(20 30 30)
LABELS=(
  "official OT-CFM exact"
  "Flash full-pixel 1K eps=0.03"
  "Flash full-pixel 2K eps=0.03"
)

download_file() {
  local url="$1"
  local out="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 --max-tries=20 --retry-wait=5 \
      --allow-overwrite=true --auto-file-renaming=false \
      -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Need aria2c, curl, or wget to download $url" >&2
    exit 1
  fi
}

verify_npz() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import sys
import numpy as np
with np.load(sys.argv[1]) as f:
    assert "mu" in f and "sigma" in f
PY
}

ensure_cleanfid_stats() {
  if [ "$DRY_RUN" = "1" ] || [ "$DOWNLOAD_CLEANFID_STATS" != "1" ]; then
    return 0
  fi

  local stats_dir
  stats_dir="$("$PYTHON_BIN" - <<'PY'
import os, cleanfid
d = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
os.makedirs(d, exist_ok=True)
print(d)
PY
)"
  echo "CleanFID stats dir: $stats_dir"

  for split in train test; do
    local file="cifar10_legacy_tensorflow_${split}_32.npz"
    local path="$stats_dir/$file"
    if [ -f "$path" ]; then
      if verify_npz "$path"; then
        echo "stats OK: $path"
        continue
      fi
      echo "removing corrupt stats: $path"
      rm -f "$path" "$path.aria2"
    fi
    echo "downloading CleanFID stats: $file"
    download_file "https://www.cs.cmu.edu/~clean-fid/stats/$file" "$path"
    verify_npz "$path"
  done

  "$PYTHON_BIN" - <<'PY'
import os, cleanfid, numpy as np
stats_dir = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
for name in ("cifar10_legacy_tensorflow_train_32.npz", "cifar10_legacy_tensorflow_test_32.npz"):
    path = os.path.join(stats_dir, name)
    with np.load(path) as f:
        print(name, f["mu"].shape, f["sigma"].shape, f"{os.path.getsize(path) / 1024 / 1024:.2f} MB")
PY
}

prepare_cifar() {
  if [ "$DRY_RUN" = "1" ] || [ "$PREPARE_CIFAR" != "1" ]; then
    return 0
  fi
  "$PYTHON_BIN" - <<PY
from torchvision.datasets import CIFAR10
root = "$DATA_DIR"
CIFAR10(root=root, train=True, download=True)
CIFAR10(root=root, train=False, download=True)
print("CIFAR-10 ready:", root)
PY
}

run_name() {
  local idx="$1"
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" \
    "${MODES[$idx]}" "${CONTEXTS[$idx]}" "${EPS_VALUES[$idx]}" "${ITERS[$idx]}" "$BATCH" "$SEED"
}

launch_train() {
  local idx="$1"
  local gpu="${GPUS[$idx]}"
  local mode="${MODES[$idx]}"
  local context="${CONTEXTS[$idx]}"
  local eps="${EPS_VALUES[$idx]}"
  local iters="${ITERS[$idx]}"
  local name
  name="$(run_name "$idx")"
  local ckpt="$OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
  local log="$LOG_DIR/gpu${gpu}_${name}.log"

  echo "=== GPU $gpu train: ${LABELS[$idx]} ==="
  echo "run=$name"
  echo "log=$log"
  if [ "$SKIP_DONE" = "1" ] && [ -f "$ckpt" ]; then
    echo "skip training: found $ckpt"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" examples/images/cifar10/train_cifar10_global_ot.py
    --data_dir "$DATA_DIR"
    --output_dir "$OUT"
    --batch_size "$BATCH"
    --total_steps "$STEPS"
    --num_workers "$NUM_WORKERS"
    --amp
    --cost_feature_dim 0
    --lr "$LR"
    --warmup "$WARMUP"
    --lr_schedule "$LR_SCHEDULE"
    --min_lr_ratio "$MIN_LR_RATIO"
    --grad_clip 1.0
    --ema_decay 0.9999
    --save_step "$SAVE_STEP"
    --sample_every "$SAMPLE_EVERY"
    --log_step "$LOG_STEP"
    --val_every "$VAL_EVERY"
    --seed "$SEED"
    --num_channel 128
    --num_res_blocks 2
    --channel_mult 1,2,2,2
    --attention_resolutions 16
    --num_heads 4
    --num_head_channels 64
    --dropout 0.1
    --pot_num_threads "$POT_NUM_THREADS"
    --coupling_mode "$mode"
    --context_size "$context"
    --eps "$eps"
    --sinkhorn_iters "$iters"
  )

  printf 'command: CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
  TRAIN_PIDS+=("$!")
  TRAIN_LABELS+=("gpu${gpu}:${name}")
}

wait_for_training() {
  local status=0
  for i in "${!TRAIN_PIDS[@]}"; do
    if wait "${TRAIN_PIDS[$i]}"; then
      echo "done training: ${TRAIN_LABELS[$i]}"
    else
      local code="$?"
      echo "failed training: ${TRAIN_LABELS[$i]} exit=$code"
      status=1
    fi
  done
  return "$status"
}

launch_eval() {
  local idx="$1"
  local split="$2"
  local num_gen="$3"
  local nfe="$4"
  local eval_dir="$5"
  local gpu="${GPUS[$idx]}"
  local name
  name="$(run_name "$idx")"
  local ckpt="$OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
  local json="$eval_dir/${name}_${split}${num_gen}_euler${nfe}.json"
  local log="$eval_dir/_logs/gpu${gpu}_${name}_${split}${num_gen}_euler${nfe}.log"

  if [ "$DRY_RUN" != "1" ] && [ ! -f "$ckpt" ]; then
    echo "missing checkpoint for eval: $ckpt" >&2
    return 1
  fi
  if [ "$SKIP_DONE" = "1" ] && [ -f "$json" ]; then
    echo "skip eval: found $json"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" examples/images/cifar10/evaluate_cifar10_global_ot.py
    --checkpoint "$ckpt"
    --out_json "$json"
    --num_gen "$num_gen"
    --batch_size_fid "$EVAL_BATCH_SIZE_FID"
    --integration_method euler
    --integration_steps "$nfe"
    --dataset_split "$split"
    --fid_mode "$FID_MODE"
    --device cuda
    --seed "$EVAL_SEED"
  )

  echo "=== GPU $gpu eval: $name split=$split num_gen=$num_gen nfe=$nfe ==="
  echo "log=$log"
  printf 'command: CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
  EVAL_PIDS+=("$!")
  EVAL_LABELS+=("gpu${gpu}:${name}:${split}${num_gen}:euler${nfe}")
}

wait_for_eval() {
  local status=0
  for i in "${!EVAL_PIDS[@]}"; do
    if wait "${EVAL_PIDS[$i]}"; then
      echo "done eval: ${EVAL_LABELS[$i]}"
    else
      local code="$?"
      echo "failed eval: ${EVAL_LABELS[$i]} exit=$code"
      status=1
    fi
  done
  return "$status"
}

summarize_eval_dir() {
  local eval_dir="$1"
  EVAL_DIR="$eval_dir" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_DIR"])
rows = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows.extend(data)
    else:
        rows.append(data)

fields = [
    "run",
    "step",
    "integration_method",
    "integration_steps",
    "fid",
    "elapsed_s",
    "num_gen",
    "seed",
    "checkpoint",
]
summary = root / "summary.csv"
with summary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {summary}")

if rows:
    print("| Method | step | split num_gen | Euler NFE | FID | minutes |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: r["run"]):
        split_size = row["num_gen"]
        print(
            f"| {row['run']} | {int(row['step'])} | {split_size} | "
            f"{int(row['integration_steps'])} | {float(row['fid']):.3f} | "
            f"{float(row.get('elapsed_s', 0.0)) / 60.0:.1f} |"
        )
PY
}

main() {
  cd "$CFM_DIR"
  mkdir -p "$DATA_DIR" "$OUT" "$LOG_DIR" "$EVAL_OUT"

  echo "=== CIFAR-10 1GPU bs128 400k train + train/test FID ==="
  echo "cfm_dir=$CFM_DIR"
  echo "repo_dir=$REPO_DIR"
  echo "data_dir=$DATA_DIR"
  echo "out=$OUT"
  echo "eval_out=$EVAL_OUT"
  echo "batch=$BATCH steps=$STEPS primary_step=$PRIMARY_STEP lr=$LR schedule=$LR_SCHEDULE"
  echo "train eval: split=train num_gen=$TRAIN_EVAL_NUM_GEN nfe=$TRAIN_EVAL_NFE"
  echo "test eval:  split=test  num_gen=$TEST_EVAL_NUM_GEN nfe=$TEST_EVAL_NFE"

  "$PYTHON_BIN" --version
  grep -n "official_otcfm_exact" examples/images/cifar10/train_cifar10_global_ot.py

  ensure_cleanfid_stats
  prepare_cifar

  TRAIN_PIDS=()
  TRAIN_LABELS=()
  if [ "$TRAIN" = "1" ]; then
    for idx in 0 1 2; do
      launch_train "$idx"
    done
    if [ "$DRY_RUN" != "1" ]; then
      wait_for_training
    fi
  fi

  if [ "$EVAL" = "1" ]; then
    local train_eval_dir="$EVAL_OUT/train50k"
    local test_eval_dir="$EVAL_OUT/test10k"
    mkdir -p "$train_eval_dir/_logs" "$test_eval_dir/_logs"

    echo "=== phase 1: train-reference CleanFID ==="
    EVAL_PIDS=()
    EVAL_LABELS=()
    for idx in 0 1 2; do
      launch_eval "$idx" train "$TRAIN_EVAL_NUM_GEN" "$TRAIN_EVAL_NFE" "$train_eval_dir"
    done
    if [ "$DRY_RUN" != "1" ]; then
      wait_for_eval
      summarize_eval_dir "$train_eval_dir"
    fi

    echo "=== phase 2: test-reference CleanFID ==="
    EVAL_PIDS=()
    EVAL_LABELS=()
    for idx in 0 1 2; do
      launch_eval "$idx" test "$TEST_EVAL_NUM_GEN" "$TEST_EVAL_NFE" "$test_eval_dir"
    done
    if [ "$DRY_RUN" != "1" ]; then
      wait_for_eval
      summarize_eval_dir "$test_eval_dir"
    fi
  fi
}

main "$@"

