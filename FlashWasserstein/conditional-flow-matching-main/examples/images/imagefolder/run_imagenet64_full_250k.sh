#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/imagenet-1k-256x256}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/data}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/imagenet64_full_250k}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-250000}"
BATCH="${BATCH:-2048}"
ACCUM="${ACCUM:-1}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-256}"
FLASH_CONTEXT="${FLASH_CONTEXT:-32768}"
METHODS="${METHODS:-independent local_exact_pot local_entropic flash_global_entropic}"

if ! compgen -G "$DATA_DIR/validation-*.parquet" >/dev/null; then
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python examples/images/imagefolder/download_hf_imagenet256.py \
    --dest "$DATASET_ROOT" \
    --split validation
fi

MODEL_FLAGS="--image_size 64 --num_channel 192 --num_res_blocks 3 --channel_mult 1,2,3,4 --attention_resolutions 32,16,8 --num_heads 4 --num_head_channels 64 --dropout 0.1 --use_checkpoint --resblock_updown --use_scale_shift_norm --use_new_attention_order"
COMMON="--data_dir $DATA_DIR --output_dir $OUT --batch_size $BATCH --grad_accum_steps $ACCUM --total_steps $STEPS --num_workers 8 --arrow_batch_size 256 --amp --amp_dtype bf16 --class_conditional --cost_feature_dim $COST_DIM --lr 1e-4 --warmup 5000 --grad_clip 1.0 --ema_decay 0.9999 --save_step 25000 --sample_every 25000 --sample_batch 64 --integration_steps 100 --log_step 20 --seed $SEED $MODEL_FLAGS"

for METHOD in $METHODS; do
  case "$METHOD" in
    independent)
      echo "=== ImageNet-64 independent FM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
        $COMMON \
        --coupling_mode independent \
        --context_size 8192 \
        --eps 0.05 \
        --sinkhorn_iters 20
      ;;
    local_exact_pot|ot_cfm|pot)
      echo "=== ImageNet-64 local exact POT OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
        $COMMON \
        --coupling_mode local_exact_pot \
        --context_size 8192 \
        --eps 0.05 \
        --sinkhorn_iters 20 \
        --pot_num_threads 1
      ;;
    local_entropic)
      echo "=== ImageNet-64 local entropic OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
        $COMMON \
        --coupling_mode local_entropic \
        --context_size 8192 \
        --eps 0.01 \
        --sinkhorn_iters 30
      ;;
    flash_global_entropic|flash|flash32768)
      echo "=== ImageNet-64 Flash global entropic OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/imagefolder/train_hf_parquet_global_ot.py \
        $COMMON \
        --coupling_mode flash_global_entropic \
        --context_size "$FLASH_CONTEXT" \
        --eps 0.01 \
        --sinkhorn_iters 30
      ;;
    *)
      echo "Unknown method '$METHOD'. Use independent, local_exact_pot, local_entropic, or flash_global_entropic." >&2
      exit 2
      ;;
  esac
done

echo "done: $OUT"
