# CIFAR-10 H100 Training Handoff

This file is a concrete handoff for running the clean CIFAR-10 FlashSinkhorn
experiments on H100 nodes. The goal is to reproduce and extend the current
winning CIFAR-10 setup faster, while avoiding the earlier exploratory configs
that were not paper-clean.

## Repository And Environment

Expected remote repository root:

```bash
cd /mindopt/ea120/temp/FlashWasserstein/conditional-flow-matching-main
```

Activate the environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate fm
```

Sanity checks:

```bash
python --version
python - <<'PY'
import torch, torchvision, cleanfid
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cleanfid", cleanfid.__file__)
PY
```

The required training script must support `official_otcfm_exact`:

```bash
grep -n "official_otcfm_exact" examples/images/cifar10/train_cifar10_global_ot.py
```

If that grep is empty, update the node code before training. The clean baseline
uses TorchCFM's `ExactOptimalTransportConditionalFlowMatcher(ot_method="exact")`;
the older `local_exact_pot` mode is not the official OT-CFM baseline.

## CIFAR-10 Dataset

Use torchvision CIFAR-10 at:

```bash
/mindopt/ea120/datasets/cifar10
```

Prepare it once:

```bash
mkdir -p /mindopt/ea120/datasets/cifar10
python - <<'PY'
from torchvision.datasets import CIFAR10
root = "/mindopt/ea120/datasets/cifar10"
CIFAR10(root=root, train=True, download=True)
CIFAR10(root=root, train=False, download=True)
print("CIFAR-10 ready")
PY
```

If the direct torchvision download is slow, download the tarball manually:

```bash
sudo apt-get update
sudo apt-get install -y curl aria2

cd /mindopt/ea120/datasets/cifar10
aria2c -x 8 -s 8 --max-tries=20 --retry-wait=5 \
  -o cifar-10-python.tar.gz \
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz

python - <<'PY'
from torchvision.datasets import CIFAR10
root = "/mindopt/ea120/datasets/cifar10"
CIFAR10(root=root, train=True, download=False)
CIFAR10(root=root, train=False, download=False)
print("CIFAR-10 ready")
PY
```

## CleanFID Reference Stats

Paper-standard CIFAR-10 FID uses:

```text
50k generated samples vs CIFAR-10 train CleanFID stats
mode = legacy_tensorflow
dataset_split = train
dataset_res = 32
```

Cache CleanFID stats before evaluation so jobs do not hang on remote downloads:

```bash
STATS_DIR=$(python - <<'PY'
import os, cleanfid
d = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
os.makedirs(d, exist_ok=True)
print(d)
PY
)

cd "$STATS_DIR"
aria2c -x 8 -s 8 --max-tries=20 --retry-wait=5 \
  --allow-overwrite=true --auto-file-renaming=false \
  -o cifar10_legacy_tensorflow_train_32.npz \
  https://www.cs.cmu.edu/~clean-fid/stats/cifar10_legacy_tensorflow_train_32.npz

python - <<'PY'
import numpy as np, os
p = "cifar10_legacy_tensorflow_train_32.npz"
print("size_MB", os.path.getsize(p) / 1024 / 1024)
with np.load(p) as f:
    print(f.files, f["mu"].shape, f["sigma"].shape)
print("train stats OK")
PY
```

Optional held-out check, not the main paper number:

```bash
aria2c -x 8 -s 8 --max-tries=20 --retry-wait=5 \
  --allow-overwrite=true --auto-file-renaming=false \
  -o cifar10_legacy_tensorflow_test_32.npz \
  https://www.cs.cmu.edu/~clean-fid/stats/cifar10_legacy_tensorflow_test_32.npz
```

If an `.npz` gives `zipfile.BadZipFile`, delete it and download again without
resume. That means an interrupted or HTML response was saved as the stats file.

## Main Training Configuration

Use full-pixel cost for the clean comparison:

```bash
--cost_feature_dim 0
```

Important cost-scale detail: `OTCouplingSampler` normalizes squared pixel cost
by `1 / (2 * feature_dim)`. For CIFAR-10 full pixels, `feature_dim = 3072`, so
CLI eps values such as `0.02` and `0.03` are already normalized eps values. Do
not multiply them by 6144 unless the code is changed to raw unscaled cost.

Use:

```text
2 GPUs per job
global batch 256
local batch 128 per GPU
200k steps
51.2M images seen
validation disabled
sample grids disabled
EMA 0.9999
```

This matches the image budget of the official single-GPU 400k-step recipe:

```text
200000 * 256 = 400000 * 128 = 51.2M images
```

The official OT-CFM baseline solves exact OT independently on each GPU over the
local `128 x 128` minibatch. FlashSinkhorn all-gathers across the two GPUs and
uses a larger target queue.

## Main H100 Training Matrix

This is the first matrix to run on an 8-GPU H100 node:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate fm
cd /mindopt/ea120/temp/FlashWasserstein/conditional-flow-matching-main

OUT=/mindopt/ea120/output/cifar10_h100_2gpu_bs256_main
mkdir -p "$OUT/_logs"

COMMON_ARGS="\
  --data_dir /mindopt/ea120/datasets/cifar10 \
  --output_dir $OUT \
  --batch_size 256 \
  --total_steps 200001 \
  --num_workers 4 \
  --amp \
  --cost_feature_dim 0 \
  --lr 2e-4 \
  --warmup 2500 \
  --grad_clip 1.0 \
  --ema_decay 0.9999 \
  --save_step 25000 \
  --sample_every 0 \
  --log_step 20 \
  --val_every 0 \
  --seed 0 \
  --num_channel 128 \
  --num_res_blocks 2 \
  --channel_mult 1,2,2,2 \
  --attention_resolutions 16 \
  --num_heads 4 \
  --num_head_channels 64 \
  --dropout 0.1 \
  --pot_num_threads 1"

# GPU 0,1: official OT-CFM exact baseline, local exact 128x128 per GPU.
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --standalone --nproc_per_node=2 \
  examples/images/cifar10/train_cifar10_global_ot.py \
  $COMMON_ARGS \
  --coupling_mode official_otcfm_exact \
  --context_size 128 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  > "$OUT/_logs/gpu01_official_otcfm_exact_bs256.log" 2>&1 &

# GPU 2,3: FlashSinkhorn 8K, eps 0.02.
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --standalone --nproc_per_node=2 \
  examples/images/cifar10/train_cifar10_global_ot.py \
  $COMMON_ARGS \
  --coupling_mode flash_global_entropic \
  --context_size 8192 \
  --eps 0.02 \
  --sinkhorn_iters 30 \
  > "$OUT/_logs/gpu23_flash8k_pixel_eps002_bs256.log" 2>&1 &

# GPU 4,5: FlashSinkhorn 8K, eps 0.03.
CUDA_VISIBLE_DEVICES=4,5 nohup torchrun --standalone --nproc_per_node=2 \
  examples/images/cifar10/train_cifar10_global_ot.py \
  $COMMON_ARGS \
  --coupling_mode flash_global_entropic \
  --context_size 8192 \
  --eps 0.03 \
  --sinkhorn_iters 30 \
  > "$OUT/_logs/gpu45_flash8k_pixel_eps003_bs256.log" 2>&1 &

# GPU 6,7: FlashSinkhorn 12K, eps 0.02.
CUDA_VISIBLE_DEVICES=6,7 nohup torchrun --standalone --nproc_per_node=2 \
  examples/images/cifar10/train_cifar10_global_ot.py \
  $COMMON_ARGS \
  --coupling_mode flash_global_entropic \
  --context_size 12288 \
  --eps 0.02 \
  --sinkhorn_iters 30 \
  > "$OUT/_logs/gpu67_flash12k_pixel_eps002_bs256.log" 2>&1 &

jobs -l
```

Monitor:

```bash
tail -f /mindopt/ea120/output/cifar10_h100_2gpu_bs256_main/_logs/*.log
```

Compact progress script:

```bash
cat > /tmp/watch_cifar_h100.py <<'PY'
import json, time
from pathlib import Path

ROOT = Path("/mindopt/ea120/output/cifar10_h100_2gpu_bs256_main")
TOTAL = 200000

print(time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"{'run':<68} {'step':>8} {'%':>6} {'step_s':>8} {'img/s':>8} {'ot_s':>8} {'eta':>9} {'loss':>10} {'memGB':>7}")
for run in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "_logs"):
    p = run / "metrics.jsonl"
    if not p.exists():
        print(f"{run.name:<68} no metrics yet")
        continue
    last = None
    with p.open() as f:
        for line in f:
            try:
                last = json.loads(line)
            except Exception:
                pass
    if last is None:
        print(f"{run.name:<68} no metrics yet")
        continue
    step = int(last["step"])
    step_s = float(last.get("step_s", last.get("step_time_s", 0.0)))
    ot_s = float(last.get("ot_s", last.get("ot_time_s", 0.0)))
    loss = float(last.get("loss", 0.0))
    mem = float(last.get("peak_mem_gb", 0.0))
    img_s = 256.0 / max(step_s, 1e-12)
    rem = max(TOTAL - step, 0) * step_s
    print(f"{run.name:<68} {step:8d} {100*step/TOTAL:5.1f}% {step_s:8.3f} {img_s:8.1f} {ot_s:8.3f} {int(rem//3600):02d}h{int((rem%3600)//60):02d}m {loss:10.6f} {mem:7.1f}")
PY

watch -n 15 python /tmp/watch_cifar_h100.py
```

## Final Evaluation

Main paper-style evaluation:

```text
CleanFID
dataset_split = train
fid_mode = legacy_tensorflow
num_gen = 50000
EMA checkpoint
Euler NFE = 25, 50, 100, 1000
```

Run one model per GPU:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate fm
cd /mindopt/ea120/temp/FlashWasserstein/conditional-flow-matching-main

RUN_ROOT=/mindopt/ea120/output/cifar10_h100_2gpu_bs256_main
EVAL_OUT=${RUN_ROOT}_eval_train50k
mkdir -p "$EVAL_OUT/_logs"

run_eval () {
  GPU="$1"
  RUN="$2"
  TAG="$3"
  (
    for NFE in 25 50 100 1000; do
      CUDA_VISIBLE_DEVICES="$GPU" python examples/images/cifar10/evaluate_cifar10_global_ot.py \
        --checkpoint "$RUN_ROOT/$RUN/weights_step_00200000.pt" \
        --out_json "$EVAL_OUT/${TAG}_euler${NFE}.json" \
        --num_gen 50000 \
        --batch_size_fid 1024 \
        --integration_method euler \
        --integration_steps "$NFE" \
        --dataset_split train \
        --fid_mode legacy_tensorflow \
        --device cuda
    done
  ) > "$EVAL_OUT/_logs/gpu${GPU}_${TAG}.log" 2>&1 &
}

run_eval 0 official_otcfm_exact_ctx128_eps0.05_it20_bs256_seed0 official_otcfm_exact
run_eval 1 flash_global_entropic_ctx8192_eps0.02_it30_bs256_seed0 flash8k_eps002
run_eval 2 flash_global_entropic_ctx8192_eps0.03_it30_bs256_seed0 flash8k_eps003
run_eval 3 flash_global_entropic_ctx12288_eps0.02_it30_bs256_seed0 flash12k_eps002

jobs -l
```

Print the final table:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/mindopt/ea120/output/cifar10_h100_2gpu_bs256_main_eval_train50k")
rows = {}
times = {}
for p in sorted(root.glob("*.json")):
    r = json.load(open(p))
    rows.setdefault(r["run"], {})[int(r["integration_steps"])] = float(r["fid"])
    times.setdefault(r["run"], {})[int(r["integration_steps"])] = float(r.get("elapsed_s", 0.0)) / 60.0

nfes = [25, 50, 100, 1000]
print("| Method | " + " | ".join(f"Euler {n}" for n in nfes) + " |")
print("|---|" + "|".join(["---:"] * len(nfes)) + "|")
for run in sorted(rows):
    vals = [f"{rows[run][n]:.3f}" if n in rows[run] else "-" for n in nfes]
    print(f"| {run} | " + " | ".join(vals) + " |")
PY
```

Optional held-out check:

```text
num_gen = 10000
dataset_split = test
NFE = 100
```

This is not the paper-standard number, but it is a useful corroborative check.

## Checkpoint-Curve Evaluation

To see FID over training time, evaluate checkpoints:

```text
25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000
```

at Euler NFE 100. Use `num_gen=50000` for the clean curve, or `num_gen=10000`
for a quick noisy curve.

## Known Results From A800 Runs

Configuration: full-pixel cost, two GPUs per job, global batch 256, local batch
128 per GPU, 200k steps, 51.2M images seen, 50k generated samples against
CIFAR-10 train CleanFID stats.

| Method | Euler 25 | Euler 50 | Euler 100 | Euler 1000 |
|---|---:|---:|---:|---:|
| Official OT-CFM exact | 7.306 | 5.739 | 4.792 | 3.926 |
| Flash 8K eps=0.02 | 7.144 | 5.335 | 4.403 | 3.628 |
| Flash 8K eps=0.03 | 7.166 | 5.219 | 4.386 | 3.672 |

Held-out check: 10k generated samples against CIFAR-10 test CleanFID stats,
Euler 100:

| Method | Test FID, 10k gen |
|---|---:|
| Official OT-CFM exact | 8.724 |
| Flash 8K eps=0.02 | 8.533 |
| Flash 8K eps=0.03 | 8.523 |
| Flash 12K eps=0.02 | 8.669 |

## Tuning Direction

Promising tuning axes:

- Flash context: `2048`, `4096`, `8192`, `12288`, `16384`.
- Eps: `0.015`, `0.02`, `0.025`, `0.03`, `0.04`.
- Sinkhorn iterations: start with `30`; try `40` only after 8K/eps is stable.

Avoid full-pixel Flash 32K under multi-GPU DDP in the current implementation. It
is extremely slow because each rank repeats a large full-pixel solve after
all-gather. 8K is currently the best proven region.

## Paper-Safe Framing

Use this wording:

```text
At matched per-GPU batch size 128 and matched total images seen, FlashSinkhorn
with an 8K full-pixel context improves CIFAR-10 FID over the official exact
OT-CFM baseline across Euler NFE values.
```

Avoid:

```text
same optimizer batch
```

when comparing two-GPU batch 256 runs against older single-GPU batch 128 runs.
Say:

```text
same per-GPU batch size and same total images seen
```

