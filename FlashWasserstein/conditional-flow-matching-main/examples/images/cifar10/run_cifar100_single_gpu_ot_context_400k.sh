#!/usr/bin/env bash
set -euo pipefail

# CIFAR-100 single-GPU OT/Flash context sweep.
# Each training worker owns one physical GPU. As soon as a worker finishes its
# training run, it immediately evaluates that checkpoint on the same GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$CFM_DIR/../.." && pwd)"

ENV_DIR="${ENV_DIR:-/nas/peter.c/file/Qwen-new/env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-/nas/peter.c/file/output}"
DATA_DIR="${DATA_DIR:-/nas/peter.c/file/datasets/cifar100}"
TRAIN_OUT="${TRAIN_OUT:-$OUT_ROOT/cifar100_single_gpu_bs128_ot_flash_context_400k}"
EVAL_OUT="${EVAL_OUT:-$OUT_ROOT/cifar100_single_gpu_bs128_ot_flash_context_400k_eval}"
LOG_DIR="${LOG_DIR:-$TRAIN_OUT/_logs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$EVAL_OUT/_logs}"

TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
PREPARE="${PREPARE:-1}"
SKIP_DONE="${SKIP_DONE:-1}"

BATCH="${BATCH:-128}"
STEPS="${STEPS:-400000}"
SAVE_STEP="${SAVE_STEP:-50000}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-0}"
LOG_STEP="${LOG_STEP:-20}"

EVAL_STEP="${EVAL_STEP:-$STEPS}"
EVAL_NFE_LIST="${EVAL_NFE_LIST:-100}"
EVAL_BATCH_SIZE_FID="${EVAL_BATCH_SIZE_FID:-1024}"
EVAL_SEED="${EVAL_SEED:-1234}"
EVAL_TEST_NUM_GEN="${EVAL_TEST_NUM_GEN:-10000}"
EVAL_TRAIN_NUM_GEN="${EVAL_TRAIN_NUM_GEN:-50000}"

export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$CFM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

activate_env() {
  if [ -d "$ENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if command -v wget >/dev/null 2>&1; then
    wget --tries=20 --waitretry=5 --retry-connrefused -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 20 --retry-delay 5 -o "$out" "$url"
  else
    echo "Need wget or curl to download $url" >&2
    exit 1
  fi
}

prepare_cifar100() {
  if [ "$PREPARE" != "1" ]; then
    return 0
  fi

  "$PYTHON_BIN" - <<PY
from torchvision.datasets import CIFAR100
root = "$DATA_DIR"
CIFAR100(root=root, train=True, download=True)
CIFAR100(root=root, train=False, download=True)
print("CIFAR-100 ready:", root)
PY
}

ensure_cleanfid_stats() {
  if [ "$PREPARE" != "1" ]; then
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

  local test_stat="$stats_dir/cifar100_clean_test_32.npz"
  if [ -f "$test_stat" ]; then
    if "$PYTHON_BIN" - "$test_stat" <<'PY' >/dev/null 2>&1
import sys, numpy as np
with np.load(sys.argv[1]) as f:
    assert f["mu"].shape == (2048,)
    assert f["sigma"].shape == (2048, 2048)
PY
    then
      echo "stats OK: $test_stat"
    else
      echo "removing corrupt stats: $test_stat"
      rm -f "$test_stat"
    fi
  fi
  if [ ! -f "$test_stat" ]; then
    download_file \
      "https://www.cs.cmu.edu/~clean-fid/stats/cifar100_clean_test_32.npz" \
      "$test_stat"
  fi

  "$PYTHON_BIN" - <<PY
import os
from pathlib import Path

import cleanfid
import numpy as np
import torch
from cleanfid import fid
from PIL import Image
from torchvision.datasets import CIFAR100

data_dir = Path("$DATA_DIR")
png_dir = data_dir / "train_png_cleanfid"
stats_dir = Path(os.path.dirname(cleanfid.__file__)) / "stats"
stat_path = stats_dir / "cifar100_train_clean_custom_na.npz"
kid_path = stats_dir / "cifar100_train_clean_custom_na_kid.npz"

if stat_path.exists():
    try:
        with np.load(stat_path) as f:
            assert f["mu"].shape == (2048,)
            assert f["sigma"].shape == (2048, 2048)
    except Exception:
        print(f"removing corrupt custom stats: {stat_path}")
        stat_path.unlink(missing_ok=True)
        kid_path.unlink(missing_ok=True)

if not stat_path.exists():
    ds = CIFAR100(root=str(data_dir), train=True, download=False)
    png_dir.mkdir(parents=True, exist_ok=True)
    bad = []
    for path in png_dir.glob("*.png"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad.append(path)
    if bad:
        print(f"removing {len(bad)} corrupt CIFAR-100 train PNGs")
        for path in bad:
            path.unlink(missing_ok=True)
    print(f"writing/verifying CIFAR-100 train PNG reference: {png_dir}")
    for idx, (img, _label) in enumerate(ds):
        path = png_dir / f"{idx:05d}.png"
        if not path.exists():
            img.save(path)
    print("making CleanFID custom stats: cifar100_train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid.make_custom_stats(
        "cifar100_train",
        fdir=str(png_dir),
        mode="clean",
        batch_size=int("$EVAL_BATCH_SIZE_FID"),
        num_workers=int("$NUM_WORKERS"),
        device=device,
        verbose=True,
    )

for name in ["cifar100_clean_test_32.npz", "cifar100_train_clean_custom_na.npz"]:
    path = stats_dir / name
    with np.load(path) as f:
        print(name, f["mu"].shape, f["sigma"].shape, f"{path.stat().st_size / 1024 / 1024:.2f} MB")
PY
}

run_dir_name() {
  local mode="$1"
  local context="$2"
  local eps="$3"
  local iters="$4"
  printf "%s_ctx%s_eps%s_it%s_bs%s_seed%s" "$mode" "$context" "$eps" "$iters" "$BATCH" "$SEED"
}

eval_checkpoint() {
  local gpu="$1"
  local label="$2"
  local checkpoint="$3"
  local split_label="$4"
  local dataset_name="$5"
  local dataset_split="$6"
  local num_gen="$7"
  local nfe="$8"

  local json="$EVAL_OUT/${label}_step${EVAL_STEP}_euler${nfe}_${split_label}${num_gen}.json"
  local log="$EVAL_LOG_DIR/gpu${gpu}_${label}_step${EVAL_STEP}_euler${nfe}_${split_label}${num_gen}.log"

  if [ "$SKIP_DONE" = "1" ] && [ -f "$json" ]; then
    echo "skip eval: found $json"
    return 0
  fi

  echo "eval gpu=$gpu label=$label split=$split_label nfe=$nfe num_gen=$num_gen"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u examples/images/cifar10/evaluate_cifar10_global_ot.py \
    --checkpoint "$checkpoint" \
    --out_json "$json" \
    --num_gen "$num_gen" \
    --batch_size_fid "$EVAL_BATCH_SIZE_FID" \
    --integration_method euler \
    --integration_steps "$nfe" \
    --dataset_name "$dataset_name" \
    --dataset_res 32 \
    --dataset_split "$dataset_split" \
    --fid_mode clean \
    --device cuda \
    --seed "$EVAL_SEED" \
    > "$log" 2>&1
}

train_then_eval() {
  local idx="$1"
  local gpu="${GPUS[$idx]}"
  local label="${LABELS[$idx]}"
  local mode="${MODES[$idx]}"
  local context="${CONTEXTS[$idx]}"
  local eps="${EPS_VALUES[$idx]}"
  local iters="${ITERS[$idx]}"
  local run_name
  run_name="$(run_dir_name "$mode" "$context" "$eps" "$iters")"
  local checkpoint="$TRAIN_OUT/$run_name/weights_step_$(printf "%08d" "$EVAL_STEP").pt"
  local log="$LOG_DIR/gpu${gpu}_${label}.log"

  echo "worker gpu=$gpu label=$label run=$run_name"
  echo "train log: $log"

  if [ "$TRAIN" = "1" ]; then
    if [ "$SKIP_DONE" = "1" ] && [ -f "$checkpoint" ]; then
      echo "skip training: found $checkpoint"
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u examples/images/cifar10/train_cifar10_global_ot.py \
        --dataset cifar100 \
        --data_dir "$DATA_DIR" \
        --output_dir "$TRAIN_OUT" \
        --coupling_mode "$mode" \
        --context_size "$context" \
        --eps "$eps" \
        --sinkhorn_iters "$iters" \
        --cost_feature_dim "$COST_DIM" \
        --batch_size "$BATCH" \
        --total_steps "$STEPS" \
        --lr "$LR" \
        --warmup "$WARMUP" \
        --grad_clip 1.0 \
        --ema_decay 0.9999 \
        --save_step "$SAVE_STEP" \
        --sample_every 0 \
        --val_every 0 \
        --log_step "$LOG_STEP" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --amp \
        --num_channel 128 \
        --num_res_blocks 2 \
        --channel_mult 1,2,2,2 \
        --attention_resolutions 16 \
        --num_heads 4 \
        --num_head_channels 64 \
        --dropout 0.1 \
        > "$log" 2>&1
    fi
  fi

  if [ "$EVAL" = "1" ]; then
    if [ ! -f "$checkpoint" ]; then
      echo "missing checkpoint for eval: $checkpoint" >&2
      return 1
    fi
    for nfe in $EVAL_NFE_LIST; do
      eval_checkpoint "$gpu" "$label" "$checkpoint" test cifar100 test "$EVAL_TEST_NUM_GEN" "$nfe"
      eval_checkpoint "$gpu" "$label" "$checkpoint" train cifar100_train custom "$EVAL_TRAIN_NUM_GEN" "$nfe"
    done
  fi
}

summarize_eval() {
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

root = Path("$EVAL_OUT")
rows = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows.extend((path.name, row) for row in data)
    else:
        rows.append((path.name, data))

for name, row in rows:
    print(
        f'{name}: FID={row["fid"]:.4f}, step={row["step"]}, '
        f'dataset={row.get("dataset_name")}/{row.get("dataset_split")}, '
        f'NFE={row["integration_steps"]}, num_gen={row["num_gen"]}'
    )
PY
}

main() {
  activate_env
  cd "$CFM_DIR"
  mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$LOG_DIR" "$EVAL_LOG_DIR"

  echo "repo=$REPO_DIR"
  echo "cfm_dir=$CFM_DIR"
  echo "env=$ENV_DIR"
  echo "data_dir=$DATA_DIR"
  echo "train_out=$TRAIN_OUT"
  echo "eval_out=$EVAL_OUT"
  echo "batch=$BATCH steps=$STEPS save_step=$SAVE_STEP lr=$LR warmup=$WARMUP cost_dim=$COST_DIM"
  echo "eval_nfe=$EVAL_NFE_LIST test_num_gen=$EVAL_TEST_NUM_GEN train_num_gen=$EVAL_TRAIN_NUM_GEN"

  prepare_cifar100
  ensure_cleanfid_stats

  GPUS=(0 1 2 3 4)
  LABELS=(official_exact independent flash2k_eps003 flash4k_eps003 flash8k_eps003)
  MODES=(official_otcfm_exact independent flash_global_entropic flash_global_entropic flash_global_entropic)
  CONTEXTS=(128 128 2048 4096 8192)
  EPS_VALUES=(0.05 0.05 0.03 0.03 0.03)
  ITERS=(20 20 30 30 30)

  pids=()
  for idx in "${!GPUS[@]}"; do
    train_then_eval "$idx" > "$LOG_DIR/worker_gpu${GPUS[$idx]}_${LABELS[$idx]}.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

  summarize_eval
  exit "$status"
}

main "$@"
