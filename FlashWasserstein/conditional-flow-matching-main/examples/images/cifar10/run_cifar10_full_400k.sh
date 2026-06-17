#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_full_400k}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-400000}"
BATCH="${BATCH:-1024}"
SEED="${SEED:-0}"
COST_DIM="${COST_DIM:-256}"
FLASH_CONTEXT="${FLASH_CONTEXT:-32768}"
METHODS="${METHODS:-independent local_exact_pot local_entropic flash_global_entropic}"

COMMON="--data_dir $DATA_DIR --output_dir $OUT --batch_size $BATCH --total_steps $STEPS --num_workers 8 --amp --cost_feature_dim $COST_DIM --lr 2e-4 --warmup 5000 --grad_clip 1.0 --ema_decay 0.9999 --save_step 50000 --sample_every 25000 --sample_batch 64 --integration_steps 100 --log_step 20 --seed $SEED --num_channel 128 --num_res_blocks 2 --channel_mult 1,2,2,2 --attention_resolutions 16 --num_heads 4 --num_head_channels 64 --dropout 0.1"

for METHOD in $METHODS; do
  case "$METHOD" in
    independent)
      echo "=== CIFAR-10 independent FM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/train_cifar10_global_ot.py \
        $COMMON \
        --coupling_mode independent \
        --context_size 8192 \
        --eps 0.05 \
        --sinkhorn_iters 20
      ;;
    local_exact_pot|ot_cfm|pot)
      echo "=== CIFAR-10 local exact POT OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/train_cifar10_global_ot.py \
        $COMMON \
        --coupling_mode local_exact_pot \
        --context_size 8192 \
        --eps 0.05 \
        --sinkhorn_iters 20 \
        --pot_num_threads 1
      ;;
    local_entropic)
      echo "=== CIFAR-10 local entropic OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/train_cifar10_global_ot.py \
        $COMMON \
        --coupling_mode local_entropic \
        --context_size 8192 \
        --eps 0.01 \
        --sinkhorn_iters 30
      ;;
    flash_global_entropic|flash|flash32768)
      echo "=== CIFAR-10 Flash global entropic OT-CFM ==="
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/train_cifar10_global_ot.py \
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
