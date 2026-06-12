#!/usr/bin/env bash
set -u

DATA_DIR="${1:-$HOME/datasets/imagenet-1k-256x256/data}"
OUT_DIR="${2:-$HOME/datasets/imagenet-1k-256x256/sdvae_latents}"
LOGDIR="${3:-$HOME/FlashWasserstein/output/latent_encode_logs}"
WORLD_SIZE="${4:-10}"
BATCH_SIZE="${5:-64}"
SHARD_SIZE="${6:-1024}"
PROJ_DIM="${7:-256}"
CALIBRATION_SAMPLES="${8:-65536}"
GPU_LIST="${GPU_LIST:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$ROOT_DIR/../env/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/../env/bin/activate"
fi

mkdir -p "$OUT_DIR" "$LOGDIR"
echo "start $(date)" > "$LOGDIR/session.log"
echo "data_dir=$DATA_DIR" >> "$LOGDIR/session.log"
echo "out_dir=$OUT_DIR" >> "$LOGDIR/session.log"

if [ -n "$GPU_LIST" ]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
  WORLD_SIZE="${#GPU_IDS[@]}"
else
  GPU_IDS=()
  for R in $(seq 0 $((WORLD_SIZE - 1))); do
    GPU_IDS+=("$R")
  done
fi
echo "world_size=$WORLD_SIZE batch_size=$BATCH_SIZE shard_size=$SHARD_SIZE" >> "$LOGDIR/session.log"
echo "gpu_list=${GPU_IDS[*]}" >> "$LOGDIR/session.log"

PIDS=()
for R in $(seq 0 $((WORLD_SIZE - 1))); do
  GPU_ID="${GPU_IDS[$R]}"
  (
    set -e
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python examples/images/latent_imagenet/encode_imagenet256_latents.py \
      --data_dir "$DATA_DIR" \
      --out_dir "$OUT_DIR" \
      --batch_size "$BATCH_SIZE" \
      --shard_size "$SHARD_SIZE" \
      --encode_rank "$R" \
      --encode_world_size "$WORLD_SIZE" \
      --skip_projection \
      --local_files_only
  ) > "$LOGDIR/encode_r${R}.log" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAIL=1
done

if [ "$FAIL" -ne 0 ]; then
  echo "encode_failed $(date)" >> "$LOGDIR/session.log"
  exit 1
fi

echo "calibration_start $(date)" >> "$LOGDIR/session.log"
(
  set -e
  cd "$ROOT_DIR"
  LAST_GPU="${GPU_IDS[$((WORLD_SIZE - 1))]}"
  CUDA_VISIBLE_DEVICES="$LAST_GPU" python examples/images/latent_imagenet/calibrate_latent_projection.py \
    --latent_dir "$OUT_DIR" \
    --proj_dim "$PROJ_DIM" \
    --calibration_samples "$CALIBRATION_SAMPLES" \
    --save_reconstruction
) > "$LOGDIR/calibrate.log" 2>&1

echo "done $(date)" >> "$LOGDIR/session.log"
