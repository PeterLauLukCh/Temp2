#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/imagenet-1k-256x256}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/data}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/imagenet128_batch_calibration}"
NPROC="${NPROC:-8}"
BATCHES="${BATCHES:-1024 1536 2048}"
STEPS="${STEPS:-200}"
CLASS_AWARE="${CLASS_AWARE:-0}"

if ! compgen -G "$DATA_DIR/validation-*.parquet" >/dev/null; then
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python examples/images/imagefolder/download_hf_imagenet256.py \
    --dest "$DATASET_ROOT" \
    --split validation
fi

MODEL_FLAGS="--image_size 128 --num_channel 256 --num_res_blocks 2 --channel_mult 1,1,2,3,4 --attention_resolutions 32,16,8 --num_heads 4 --num_head_channels -1 --dropout 0.0 --use_checkpoint --resblock_updown --use_scale_shift_norm --use_new_attention_order"
CLASS_AWARE_FLAG=""
if [ "$CLASS_AWARE" = "1" ]; then
  CLASS_AWARE_FLAG="--class_aware_coupling"
fi
COMMON="--data_dir $DATA_DIR --output_dir $OUT --total_steps $STEPS --num_workers 4 --arrow_batch_size 128 --amp --amp_dtype bf16 --class_conditional $CLASS_AWARE_FLAG --cost_feature_dim 512 --lr 1e-4 --warmup 100 --save_step 0 --sample_every 0 --log_step 10 $MODEL_FLAGS"

for BATCH in $BATCHES; do
  case "$BATCH" in
    512) ACCUM=4 ;;
    768) ACCUM=3 ;;
    1024) ACCUM=2 ;;
    1536) ACCUM=2 ;;
    2048) ACCUM=1 ;;
    *) ACCUM=1 ;;
  esac
  echo "=== flash calibration batch=$BATCH accum=$ACCUM effective=$((BATCH * ACCUM)) ==="
  torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
    $COMMON \
    --batch_size "$BATCH" \
    --grad_accum_steps "$ACCUM" \
    --coupling_mode flash_global_entropic \
    --context_size 32768 \
    --eps 0.01 \
    --sinkhorn_iters 30
done

echo "done: $OUT"
