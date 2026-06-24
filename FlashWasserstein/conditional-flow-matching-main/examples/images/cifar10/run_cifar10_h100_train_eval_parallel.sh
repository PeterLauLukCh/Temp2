#!/usr/bin/env bash
set -euo pipefail

# H100 CIFAR-10 launcher for the clean full-pixel FlashSinkhorn comparison.
#
# Default target layout:
#   env:  /nas/peter.c/file/Qwen-new/env
#   repo: auto-detected from this script location
#   eval: /nas/peter.c/file/output/cifar10_h100_2gpu_bs256_main_eval_train50k
#
# Examples:
#   bash examples/images/cifar10/run_cifar10_h100_train_eval_parallel.sh
#
#   TRAIN_GPU_GROUPS="0,1" EVAL_GPU_LIST="0 1" \
#     bash examples/images/cifar10/run_cifar10_h100_train_eval_parallel.sh
#
#   TRAIN=0 EVAL=1 EVAL_NFE_LIST="100" EVAL_GPU_LIST="0 1" \
#     bash examples/images/cifar10/run_cifar10_h100_train_eval_parallel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$CFM_DIR/../.." && pwd)"

ENV_DIR="${ENV_DIR:-/nas/peter.c/file/Qwen-new/env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-/nas/peter.c/file/output}"
DATA_DIR="${DATA_DIR:-/nas/peter.c/file/datasets/cifar10}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_ROOT/cifar10_h100_2gpu_bs256_main}"
EVAL_OUT="${EVAL_OUT:-$OUT_ROOT/cifar10_h100_2gpu_bs256_main_eval_train50k}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$TRAIN_OUT/_logs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$EVAL_OUT/_logs}"

TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
PREPARE_CIFAR="${PREPARE_CIFAR:-1}"
DOWNLOAD_CLEANFID_STATS="${DOWNLOAD_CLEANFID_STATS:-1}"
CLEANFID_SPLITS="${CLEANFID_SPLITS:-train test}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

# Semicolon-separated groups. Each group is exposed as CUDA_VISIBLE_DEVICES for
# one DDP training job. A single group such as "0,1" runs the matrix in waves.
TRAIN_GPU_GROUPS="${TRAIN_GPU_GROUPS:-0,1;2,3;4,5;6,7}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-}"

BATCH="${BATCH:-256}"
STEPS="${STEPS:-200001}"
PRIMARY_STEP="${PRIMARY_STEP:-200000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-0}"
SAVE_STEP="${SAVE_STEP:-25000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"
LOG_STEP="${LOG_STEP:-20}"
VAL_EVERY="${VAL_EVERY:-0}"
POT_NUM_THREADS="${POT_NUM_THREADS:-1}"

EVAL_STEPS_LIST="${EVAL_STEPS_LIST:-$PRIMARY_STEP}"
EVAL_NFE_LIST="${EVAL_NFE_LIST:-25 50 100 1000}"
EVAL_NUM_GEN="${EVAL_NUM_GEN:-50000}"
EVAL_BATCH_SIZE_FID="${EVAL_BATCH_SIZE_FID:-1024}"
EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-train}"
EVAL_FID_MODE="${EVAL_FID_MODE:-legacy_tensorflow}"
EVAL_SEED="${EVAL_SEED:-1234}"
COMPUTE_IS="${COMPUTE_IS:-0}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-512}"
IS_SPLITS="${IS_SPLITS:-10}"

# Select configs by zero-based index from the matrix below.
# 0: official OT-CFM exact
# 1: Flash 8K eps=0.02
# 2: Flash 8K eps=0.03
# 3: Flash 12K eps=0.02
RUN_INDEXES="${RUN_INDEXES:-0 1 2 3}"
EVAL_RUNS="${EVAL_RUNS:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$CFM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

MODES=(
  official_otcfm_exact
  flash_global_entropic
  flash_global_entropic
  flash_global_entropic
)
CONTEXTS=(128 8192 8192 12288)
EPS_VALUES=(0.05 0.02 0.03 0.02)
ITERS=(20 30 30 30)
LABELS=(
  "official OT-CFM exact"
  "Flash 8K pixel eps=0.02"
  "Flash 8K pixel eps=0.03"
  "Flash 12K pixel eps=0.02"
)

activate_env() {
  if [ -d "$ENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
  fi
}

count_gpus() {
  local group="$1"
  local spaced="${group//,/ }"
  # shellcheck disable=SC2086
  set -- $spaced
  echo "$#"
}

group_to_log_tag() {
  echo "$1" | tr ',' '-'
}

default_eval_gpus() {
  echo "$TRAIN_GPU_GROUPS" | tr ';,' ' ' | awk '
    NF {
      for (i = 1; i <= NF; i++) {
        if (!seen[$i]++) {
          printf "%s%s", sep, $i
          sep = " "
        }
      }
    }
    END { print "" }
  '
}

run_name_for_index() {
  local idx="$1"
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" \
    "${MODES[$idx]}" "${CONTEXTS[$idx]}" "${EPS_VALUES[$idx]}" "${ITERS[$idx]}" "$BATCH" "$SEED"
}

download_file() {
  local url="$1"
  local out="$2"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  else
    echo "Need wget or curl to download $url" >&2
    exit 1
  fi
}

ensure_cleanfid_stats() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: skip CleanFID stats download/check"
    return 0
  fi
  if [ "$DOWNLOAD_CLEANFID_STATS" != "1" ]; then
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
  for split in $CLEANFID_SPLITS; do
    local file="cifar10_legacy_tensorflow_${split}_32.npz"
    local path="$stats_dir/$file"
    if [ -f "$path" ]; then
      if "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import sys, numpy as np
with np.load(sys.argv[1]) as f:
    assert "mu" in f and "sigma" in f
PY
      then
        echo "stats OK: $path"
        continue
      fi
      echo "removing corrupt stats: $path"
      rm -f "$path"
    fi
    echo "downloading $file"
    download_file "https://www.cs.cmu.edu/~clean-fid/stats/$file" "$path"
  done
  "$PYTHON_BIN" - <<'PY'
import os, cleanfid, numpy as np
stats_dir = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
for name in sorted(p for p in os.listdir(stats_dir) if p.startswith("cifar10_legacy_tensorflow_")):
    path = os.path.join(stats_dir, name)
    with np.load(path) as f:
        print(name, f["mu"].shape, f["sigma"].shape, f"{os.path.getsize(path) / 1024 / 1024:.2f} MB")
PY
}

prepare_cifar() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: skip CIFAR-10 preparation"
    return 0
  fi
  if [ "$PREPARE_CIFAR" != "1" ]; then
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

launch_train() {
  local idx="$1"
  local group="$2"
  local mode="${MODES[$idx]}"
  local context="${CONTEXTS[$idx]}"
  local eps="${EPS_VALUES[$idx]}"
  local iters="${ITERS[$idx]}"
  local name
  name="$(run_name_for_index "$idx")"
  local done_ckpt="$TRAIN_OUT/$name/weights_step_$(printf "%08d" "$PRIMARY_STEP").pt"
  local nproc
  nproc="$(count_gpus "$group")"
  local group_tag
  group_tag="$(group_to_log_tag "$group")"
  local log="$TRAIN_LOG_DIR/gpu${group_tag}_${name}.log"

  echo "=== train group=$group: ${LABELS[$idx]} ==="
  echo "run=$name"
  echo "log=$log"
  if [ "$SKIP_DONE" = "1" ] && [ -f "$done_ckpt" ]; then
    echo "skip training: found $done_ckpt"
    return 0
  fi

  local cmd=(
    torchrun --standalone --nproc_per_node="$nproc"
    examples/images/cifar10/train_cifar10_global_ot.py
    --data_dir "$DATA_DIR"
    --output_dir "$TRAIN_OUT"
    --batch_size "$BATCH"
    --total_steps "$STEPS"
    --num_workers "$NUM_WORKERS"
    --amp
    --cost_feature_dim "$COST_DIM"
    --lr 2e-4
    --warmup 2500
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

  printf 'command: CUDA_VISIBLE_DEVICES=%q' "$group"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$group" "${cmd[@]}" >"$log" 2>&1 &
  TRAIN_PIDS+=("$!")
  TRAIN_PID_LABELS+=("group${group_tag}:${name}")
}

run_training_matrix() {
  read -r -a groups <<< "$(echo "$TRAIN_GPU_GROUPS" | tr ';' ' ')"
  read -r -a selected <<< "$RUN_INDEXES"
  if [ "${#groups[@]}" -eq 0 ]; then
    echo "TRAIN_GPU_GROUPS is empty" >&2
    exit 1
  fi
  if [ "${#selected[@]}" -eq 0 ]; then
    echo "RUN_INDEXES is empty" >&2
    exit 1
  fi

  local wave_size="${#groups[@]}"
  local offset
  for ((offset = 0; offset < ${#selected[@]}; offset += wave_size)); do
    TRAIN_PIDS=()
    TRAIN_PID_LABELS=()
    local slot
    for ((slot = 0; slot < wave_size && offset + slot < ${#selected[@]}; slot++)); do
      local idx="${selected[$((offset + slot))]}"
      if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#MODES[@]}" ]; then
        echo "invalid RUN_INDEXES entry: $idx" >&2
        exit 1
      fi
      launch_train "$idx" "${groups[$slot]}"
    done

    if [ "$DRY_RUN" = "1" ]; then
      continue
    fi

    local status=0
    for i in "${!TRAIN_PIDS[@]}"; do
      if wait "${TRAIN_PIDS[$i]}"; then
        echo "done training: ${TRAIN_PID_LABELS[$i]}"
      else
        local code="$?"
        echo "failed training: ${TRAIN_PID_LABELS[$i]} exit=$code"
        status=1
      fi
    done
    if [ "$status" != "0" ]; then
      echo "training failed; skipping evaluation"
      exit "$status"
    fi
  done
}

build_eval_run_list() {
  if [ -n "$EVAL_RUNS" ]; then
    read -r -a EVAL_RUN_NAMES <<< "$EVAL_RUNS"
    return 0
  fi
  EVAL_RUN_NAMES=()
  read -r -a selected <<< "$RUN_INDEXES"
  for idx in "${selected[@]}"; do
    EVAL_RUN_NAMES+=("$(run_name_for_index "$idx")")
  done
}

launch_eval_task() {
  local gpu="$1"
  local run="$2"
  local step="$3"
  local nfe="$4"
  local step_tag
  step_tag="$(printf "%08d" "$step")"
  local checkpoint="$TRAIN_OUT/$run/weights_step_${step_tag}.pt"
  local out_json="$EVAL_OUT/${run}_step_${step}_euler${nfe}.json"
  local log="$EVAL_LOG_DIR/gpu${gpu}_${run}_step${step}_euler${nfe}.log"

  if [ "$DRY_RUN" != "1" ] && [ ! -f "$checkpoint" ]; then
    echo "skip eval: missing $checkpoint"
    return 0
  fi
  if [ "$DRY_RUN" != "1" ] && [ "$SKIP_DONE" = "1" ] && [ -f "$out_json" ]; then
    echo "skip eval: found $out_json"
    return 0
  fi

  echo "=== eval gpu=$gpu run=$run step=$step euler_nfe=$nfe ==="
  echo "log=$log"
  local cmd=(
    "$PYTHON_BIN" examples/images/cifar10/evaluate_cifar10_global_ot.py
    --checkpoint "$checkpoint"
    --out_json "$out_json"
    --num_gen "$EVAL_NUM_GEN"
    --batch_size_fid "$EVAL_BATCH_SIZE_FID"
    --integration_method euler
    --integration_steps "$nfe"
    --dataset_split "$EVAL_DATASET_SPLIT"
    --fid_mode "$EVAL_FID_MODE"
    --device cuda
    --seed "$EVAL_SEED"
  )
  if [ "$COMPUTE_IS" = "1" ]; then
    cmd+=(--compute_is --is_batch_size "$IS_BATCH_SIZE" --is_splits "$IS_SPLITS")
  fi

  printf 'command: CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
  EVAL_PIDS+=("$!")
  EVAL_PID_LABELS+=("gpu${gpu}:${run}:step${step}:euler${nfe}")
}

run_parallel_eval() {
  if [ -z "$EVAL_GPU_LIST" ]; then
    EVAL_GPU_LIST="$(default_eval_gpus)"
  fi
  read -r -a eval_gpus <<< "$EVAL_GPU_LIST"
  if [ "${#eval_gpus[@]}" -eq 0 ]; then
    echo "EVAL_GPU_LIST is empty" >&2
    exit 1
  fi

  build_eval_run_list
  read -r -a eval_steps <<< "$EVAL_STEPS_LIST"
  read -r -a eval_nfes <<< "$EVAL_NFE_LIST"

  TASK_RUNS=()
  TASK_STEPS=()
  TASK_NFES=()
  for run in "${EVAL_RUN_NAMES[@]}"; do
    for step in "${eval_steps[@]}"; do
      for nfe in "${eval_nfes[@]}"; do
        TASK_RUNS+=("$run")
        TASK_STEPS+=("$step")
        TASK_NFES+=("$nfe")
      done
    done
  done

  echo "=== parallel eval scheduler ==="
  echo "eval_gpus=$EVAL_GPU_LIST"
  echo "eval_runs=${EVAL_RUN_NAMES[*]}"
  echo "eval_steps=$EVAL_STEPS_LIST"
  echo "eval_nfe=$EVAL_NFE_LIST"
  echo "eval_tasks=${#TASK_RUNS[@]}"

  local wave_size="${#eval_gpus[@]}"
  local offset
  for ((offset = 0; offset < ${#TASK_RUNS[@]}; offset += wave_size)); do
    EVAL_PIDS=()
    EVAL_PID_LABELS=()
    local slot
    for ((slot = 0; slot < wave_size && offset + slot < ${#TASK_RUNS[@]}; slot++)); do
      launch_eval_task \
        "${eval_gpus[$slot]}" \
        "${TASK_RUNS[$((offset + slot))]}" \
        "${TASK_STEPS[$((offset + slot))]}" \
        "${TASK_NFES[$((offset + slot))]}"
    done

    if [ "$DRY_RUN" = "1" ]; then
      continue
    fi

    local status=0
    for i in "${!EVAL_PIDS[@]}"; do
      if wait "${EVAL_PIDS[$i]}"; then
        echo "done eval: ${EVAL_PID_LABELS[$i]}"
      else
        local code="$?"
        echo "failed eval: ${EVAL_PID_LABELS[$i]} exit=$code"
        status=1
      fi
    done
    if [ "$status" != "0" ]; then
      exit "$status"
    fi
  done

  if [ "$DRY_RUN" != "1" ]; then
    summarize_eval
  fi
}

summarize_eval() {
  EVAL_OUT="$EVAL_OUT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["EVAL_OUT"])
rows = []
for path in sorted(out.glob("*.json")):
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
summary = out / "summary.csv"
with summary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {summary}")

table = {}
for row in rows:
    table.setdefault(row["run"], {})[int(row["integration_steps"])] = float(row["fid"])
nfes = sorted({int(row["integration_steps"]) for row in rows})
if nfes:
    print("| Method | " + " | ".join(f"Euler {n}" for n in nfes) + " |")
    print("|---|" + "|".join(["---:"] * len(nfes)) + "|")
    for run in sorted(table):
        vals = [f"{table[run][n]:.3f}" if n in table[run] else "-" for n in nfes]
        print(f"| {run} | " + " | ".join(vals) + " |")
PY
}

main() {
  activate_env
  cd "$CFM_DIR"
  mkdir -p "$OUT_ROOT" "$DATA_DIR" "$TRAIN_OUT" "$TRAIN_LOG_DIR" "$EVAL_OUT" "$EVAL_LOG_DIR"

  echo "=== CIFAR-10 H100 train then parallel eval ==="
  echo "repo=$REPO_DIR"
  echo "cfm_dir=$CFM_DIR"
  echo "env=$ENV_DIR"
  echo "data_dir=$DATA_DIR"
  echo "train_out=$TRAIN_OUT"
  echo "eval_out=$EVAL_OUT"
  echo "train_gpu_groups=$TRAIN_GPU_GROUPS"
  echo "eval_gpu_list=${EVAL_GPU_LIST:-auto from train groups}"
  echo "run_indexes=$RUN_INDEXES"
  echo "batch=$BATCH steps=$STEPS primary_step=$PRIMARY_STEP cost_dim=$COST_DIM"
  echo "eval_num_gen=$EVAL_NUM_GEN eval_nfe=$EVAL_NFE_LIST"

  "$PYTHON_BIN" --version
  grep -n "official_otcfm_exact" examples/images/cifar10/train_cifar10_global_ot.py
  ensure_cleanfid_stats
  prepare_cifar

  if [ "$TRAIN" = "1" ]; then
    run_training_matrix
  fi
  if [ "$EVAL" = "1" ]; then
    run_parallel_eval
  fi
}

main "$@"
