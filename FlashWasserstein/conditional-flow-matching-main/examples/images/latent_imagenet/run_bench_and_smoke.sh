#!/usr/bin/env bash
set -euo pipefail

LATENT_DIR="${1:-$HOME/datasets/imagenet-1k-256x256/sdvae_latents}"
OUT_ROOT="${2:-$HOME/FlashWasserstein/output/latent_ot_runs}"
GPU="${3:-9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$ROOT_DIR/../env/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/../env/bin/activate"
fi

if [ ! -f "$LATENT_DIR/projection.pt" ]; then
  echo "Missing $LATENT_DIR/projection.pt; wait for run_parallel_encode.sh to finish." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python examples/images/latent_imagenet/bench_latent_ot_coupling.py \
  --latent_dir "$LATENT_DIR" \
  --context_sizes 512,1024,2048,4096,8192,16384 \
  --local_batch 128 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --pot_max_context 2048 \
  --dense_max_context 4096 \
  --out_dir "$OUT_ROOT/bench"

for MODE in independent local_pot_exact_row; do
  CUDA_VISIBLE_DEVICES="$GPU" python examples/images/latent_imagenet/train_latent_imagenet.py \
    --latent_dir "$LATENT_DIR" \
    --coupling_mode "$MODE" \
    --batch_size 128 \
    --total_steps "${SMOKE_STEPS:-2000}" \
    --num_workers 4 \
    --sample_every 1000 \
    --save_step 1000 \
    --log_step 20 \
    --output_dir "$OUT_ROOT/smoke_train"
done

CUDA_VISIBLE_DEVICES="$GPU" python examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir "$LATENT_DIR" \
  --coupling_mode global_flash_sinkhorn \
  --batch_size 128 \
  --context_size "${FLASH_CONTEXT:-8192}" \
  --eps 0.05 \
  --sinkhorn_iters "${FLASH_ITERS:-80}" \
  --total_steps "${FLASH_SMOKE_STEPS:-200}" \
  --num_workers 4 \
  --sample_every 100 \
  --save_step 100 \
  --log_step 10 \
  --output_dir "$OUT_ROOT/smoke_train"
