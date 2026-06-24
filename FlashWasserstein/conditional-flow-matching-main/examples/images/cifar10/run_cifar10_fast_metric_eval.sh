#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-$HOME/FlashSinkhorn/output/cifar10_matrix}"
OUT="${OUT:-${RUN_ROOT}_fast_eval}"
LOG_DIR="${LOG_DIR:-$OUT/_logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEPS_LIST="${STEPS_LIST:-400000}"
NFE_LIST="${NFE_LIST:-10 20 50 100 1000}"
NUM_GEN="${NUM_GEN:-50000}"
BATCH_SIZE_FID="${BATCH_SIZE_FID:-1024}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
INCLUDE="${INCLUDE:-}"
FID_MODE="${FID_MODE:-legacy_tensorflow}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
COMPUTE_IS="${COMPUTE_IS:-0}"
IS_REQUIRED="${IS_REQUIRED:-0}"
IS_BATCH_SIZE="${IS_BATCH_SIZE:-512}"
IS_SPLITS="${IS_SPLITS:-10}"
SEED="${SEED:-1234}"
SKIP_DONE="${SKIP_DONE:-1}"
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
"$PYTHON_BIN" - <<'PY'
try:
    import cleanfid  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit("Missing clean-fid. Install it in the active env with: pip install clean-fid") from exc
PY
fi

if [ "$COMPUTE_IS" = "1" ] && [ "$DRY_RUN" != "1" ]; then
  if ! "$PYTHON_BIN" - <<'PY'
try:
    from torchvision.models import Inception_V3_Weights, inception_v3  # noqa: F401
except Exception as exc:
    raise SystemExit(str(exc)) from exc
PY
  then
    if [ "$IS_REQUIRED" = "1" ]; then
      echo "Inception Score requested but torchvision InceptionV3 is unavailable."
      exit 1
    fi
    echo "warning: Inception Score unavailable; continuing with FID only."
    COMPUTE_IS=0
  fi
fi

echo "=== CIFAR-10 fast metric eval ==="
echo "run_root=$RUN_ROOT"
echo "out=$OUT"
echo "steps=$STEPS_LIST"
echo "nfe=$NFE_LIST"
echo "num_gen=$NUM_GEN batch_size_fid=$BATCH_SIZE_FID"
echo "gpu_list=$GPU_LIST"
echo "fid_mode=$FID_MODE dataset_split=$DATASET_SPLIT"
echo "compute_is=$COMPUTE_IS is_batch_size=$IS_BATCH_SIZE is_splits=$IS_SPLITS"

RUN_DIRS=()
while IFS= read -r run_dir; do
  RUN_DIRS+=("$run_dir")
done < <(
RUN_ROOT="$RUN_ROOT" INCLUDE="$INCLUDE" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"]).expanduser()
needles = [item.strip() for item in os.environ.get("INCLUDE", "").split(",") if item.strip()]
for path in sorted(p for p in root.iterdir() if p.is_dir()):
    if not needles or any(needle in path.name for needle in needles):
        print(path)
PY
)

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "no run directories matched under $RUN_ROOT"
  exit 1
fi

read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty"
  exit 1
fi

declare -a PIDS=()
declare -a LABELS=()

launch_run_eval() {
  local gpu="$1"
  local run_dir="$2"
  local step="$3"
  local run_name
  run_name="$(basename "$run_dir")"
  local step_tag
  step_tag="$(printf "%08d" "$step")"
  local checkpoint="$run_dir/weights_step_${step_tag}.pt"
  local log="$LOG_DIR/gpu${gpu}_${run_name}_step${step}.log"

  if [ ! -f "$checkpoint" ]; then
    echo "skip $run_name step=$step: missing $checkpoint"
    return 0
  fi

  echo "=== GPU $gpu eval: $run_name step=$step ==="
  echo "checkpoint=$checkpoint"
  echo "log=$log"
  (
    set -euo pipefail
    for nfe in $NFE_LIST; do
      out_json="$OUT/${run_name}_step_${step}_euler${nfe}.json"
      if [ "$SKIP_DONE" = "1" ] && [ -f "$out_json" ]; then
        echo "skip existing $out_json"
        continue
      fi
      cmd=(
        "$PYTHON_BIN" examples/images/cifar10/evaluate_cifar10_global_ot.py
        --checkpoint "$checkpoint"
        --out_json "$out_json"
        --num_gen "$NUM_GEN"
        --batch_size_fid "$BATCH_SIZE_FID"
        --integration_method euler
        --integration_steps "$nfe"
        --dataset_split "$DATASET_SPLIT"
        --fid_mode "$FID_MODE"
        --device cuda
        --seed "$SEED"
      )
      if [ "$COMPUTE_IS" = "1" ]; then
        cmd+=(--compute_is --is_batch_size "$IS_BATCH_SIZE" --is_splits "$IS_SPLITS")
      fi
      printf 'command: CUDA_VISIBLE_DEVICES=%q' "$gpu"
      printf ' %q' "${cmd[@]}"
      printf '\n'
      if [ "$DRY_RUN" = "1" ]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    done
  ) >"$log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("gpu${gpu}:${run_name}:step${step}")
}

for step in $STEPS_LIST; do
  for idx in "${!RUN_DIRS[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    launch_run_eval "$gpu" "${RUN_DIRS[$idx]}" "$step"
  done
done

if [ "$DRY_RUN" = "1" ]; then
  echo "dry run complete"
  exit 0
fi

status=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "done: ${LABELS[$idx]}"
  else
    code="$?"
    echo "failed: ${LABELS[$idx]} exit=$code"
    status=1
  fi
done

"$PYTHON_BIN" - <<'PY'
import csv
import json
from pathlib import Path

out = Path(__import__("os").environ["OUT"])
rows = []
for path in sorted(out.glob("*.json")):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows.extend(data)
    else:
        rows.append(data)

summary = out / "summary.csv"
fields = [
    "run",
    "step",
    "integration_steps",
    "fid",
    "elapsed_s",
    "inception_score_mean",
    "inception_score_std",
    "inception_score_elapsed_s",
    "num_gen",
    "seed",
    "checkpoint",
]
with summary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {summary}")
PY

exit "$status"
