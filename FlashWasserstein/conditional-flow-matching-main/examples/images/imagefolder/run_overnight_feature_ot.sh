#!/usr/bin/env bash
set -euo pipefail

BASE="/home/admin-columbia/FlashWasserstein"
WORK="$BASE/conditional-flow-matching-main/examples/images/imagefolder"
DATA="/home/admin-columbia/datasets/imagenet-1k-256x256/data"
PY="$BASE/env/bin/python"
PYTHONPATH_VALUE="$BASE:$BASE/code/src:$BASE/conditional-flow-matching-main"
LOG_DIR="$WORK/overnight_logs"

mkdir -p "$LOG_DIR"
cd "$WORK"

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  echo "launch ${name} gpu=${gpu} log=${log}"
  PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES="$gpu" \
    setsid "$PY" bench_global_vs_local_feature_ot.py "$@" > "$log" 2>&1 &
  echo "$!" > "$LOG_DIR/${name}.pid"
}

COMMON=(
  --data_dir "$DATA"
  --image_size 256
  --num_blocks 10
  --seed 0
  --source gaussian
  --batch_sizes 1280,2560,5120,10240,15360
)

launch resnet50_std_gaussian 0 \
  "${COMMON[@]}" \
  --feature_mode resnet50 \
  --feature_norm standardize \
  --resnet_batch 256 \
  --out ./overnight_resnet50_std_gaussian

launch lowfreq32_std_gaussian 1 \
  "${COMMON[@]}" \
  --feature_mode lowfreq \
  --feature_size 32 \
  --feature_norm standardize \
  --out ./overnight_lowfreq32_std_gaussian

launch randproj256_from64_std_gaussian 2 \
  "${COMMON[@]}" \
  --feature_mode random_projection \
  --feature_size 64 \
  --projection_dim 256 \
  --feature_norm standardize \
  --out ./overnight_randproj256_from64_std_gaussian

launch resnet50_std_shuffled_data 3 \
  --data_dir "$DATA" \
  --image_size 256 \
  --num_blocks 10 \
  --seed 0 \
  --source shuffled_data \
  --batch_sizes 1280,2560,5120,10240 \
  --feature_mode resnet50 \
  --feature_norm standardize \
  --resnet_batch 256 \
  --out ./overnight_resnet50_std_shuffled_data

echo "started"
