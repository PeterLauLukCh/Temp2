#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/imagenet-1k-256x256}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/data}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/imagenet128_eps_preflight}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-1000}"
BATCH="${BATCH:-1024}"
ACCUM="${ACCUM:-2}"
EPS_LIST="${EPS_LIST:-0.02 0.01 0.005}"

if ! compgen -G "$DATA_DIR/validation-*.parquet" >/dev/null; then
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python examples/images/imagefolder/download_hf_imagenet256.py \
    --dest "$DATASET_ROOT" \
    --split validation
fi

MODEL_FLAGS="--image_size 128 --num_channel 256 --num_res_blocks 2 --channel_mult 1,2,3,4 --attention_resolutions 32,16,8 --num_heads 4 --num_head_channels -1 --dropout 0.0 --use_checkpoint --resblock_updown --use_scale_shift_norm --use_new_attention_order"
COMMON="--data_dir $DATA_DIR --output_dir $OUT --batch_size $BATCH --grad_accum_steps $ACCUM --total_steps $STEPS --num_workers 4 --arrow_batch_size 128 --amp --amp_dtype bf16 --class_conditional --class_aware_coupling --cost_feature_dim 512 --lr 1e-4 --warmup 100 --save_step 0 --sample_every 0 --log_step 10 $MODEL_FLAGS"

for EPS in $EPS_LIST; do
  echo "=== Flash epsilon preflight eps=$EPS ==="
  torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
    $COMMON \
    --coupling_mode flash_global_entropic \
    --context_size 32768 \
    --eps "$EPS" \
    --sinkhorn_iters 30
done

echo "done: $OUT"
