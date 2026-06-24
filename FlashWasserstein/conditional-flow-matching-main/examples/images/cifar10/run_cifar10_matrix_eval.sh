#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/cifar10_matrix_200k}"
OUT="${OUT:-$HOME/FlashSinkhorn/output/cifar10_matrix_200k_eval}"
DATA_DIR="${DATA_DIR:-$HOME/datasets/cifar10}"
LOG_DIR="${LOG_DIR:-$OUT/_logs}"
STEPS_LIST="${STEPS_LIST:-100000 200000}"
NFE_LIST="${NFE_LIST:-10 20 50 100}"
NUM_GEN="${NUM_GEN:-50000}"
NPROC="${NPROC:-8}"
BATCH_PER_GPU="${BATCH_PER_GPU:-512}"
INCLUDE="${INCLUDE:-}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"
SCORE_REFERENCE="${SCORE_REFERENCE:-cleanfid_stats}"
KID_REFERENCE="${KID_REFERENCE:-folder}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
DATASET_RES="${DATASET_RES:-32}"
ONLY_GENERATE="${ONLY_GENERATE:-0}"
ONLY_SCORE="${ONLY_SCORE:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/FlashWasserstein:$REPO_DIR/code/src:$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT_DIR"
mkdir -p "$OUT" "$LOG_DIR"

if [ "$DRY_RUN" != "1" ]; then
python - <<'PY'
try:
    import cleanfid  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit("Missing clean-fid. Install it in the active env with: pip install clean-fid") from exc
PY
fi

echo "=== CIFAR-10 matrix eval sweep ==="
echo "run_root=$RUN_ROOT"
echo "out=$OUT"
echo "data_dir=$DATA_DIR"
echo "steps=$STEPS_LIST"
echo "nfe=$NFE_LIST"
echo "num_gen=$NUM_GEN nproc=$NPROC batch_per_gpu=$BATCH_PER_GPU"
echo "score_reference=$SCORE_REFERENCE kid_reference=$KID_REFERENCE fid_mode=$FID_MODE"

for STEP in $STEPS_LIST; do
  for NFE in $NFE_LIST; do
    log="$LOG_DIR/eval_step_${STEP}_euler${NFE}.log"
    echo "=== eval step=$STEP Euler NFE=$NFE ==="
    echo "log=$log"
    cmd=(
      torchrun --standalone --nproc_per_node="$NPROC" examples/images/cifar10/evaluate_cifar10_folders.py
      --run_root "$RUN_ROOT" \
      --step "$STEP" \
      --out_dir "$OUT" \
      --data_dir "$DATA_DIR" \
      --split train \
      --num_gen "$NUM_GEN" \
      --batch_size "$BATCH_PER_GPU" \
      --integration_method euler \
      --integration_steps "$NFE" \
      --score_reference "$SCORE_REFERENCE" \
      --kid_reference "$KID_REFERENCE" \
      --fid_mode "$FID_MODE" \
      --dataset_name "$DATASET_NAME" \
      --dataset_res "$DATASET_RES" \
      --compute_kid
    )
    if [ "$AMP" = "1" ]; then
      cmd+=(--amp --amp_dtype "$AMP_DTYPE")
    fi
    if [ "$ONLY_GENERATE" = "1" ]; then
      cmd+=(--only_generate)
    fi
    if [ "$ONLY_SCORE" = "1" ]; then
      cmd+=(--only_score)
    fi
    if [ -n "$INCLUDE" ]; then
      cmd+=(--include "$INCLUDE")
    fi
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [ "$DRY_RUN" = "1" ]; then
      continue
    fi
    "${cmd[@]}" 2>&1 | tee "$log"
  done
done

echo "done: $OUT"
