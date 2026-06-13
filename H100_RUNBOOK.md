# H100 Runbook: Latent OT-CFM vs Flash Global OT-CFM

This repo contains the FlashSinkhorn code, the FlashWasserstein prototype, and
the latent ImageNet OT-CFM experiment under:

```text
FlashWasserstein/conditional-flow-matching-main/examples/images/latent_imagenet/
```

## 1. Install

```bash
git clone git@github.com:PeterLauLukChen/temp.git FlashSinkhorn
cd FlashSinkhorn/FlashWasserstein/conditional-flow-matching-main

python3 -m venv ../env
source ../env/bin/activate

pip install -U pip wheel setuptools
pip install -e .
pip install -e ../../code
pip install pytest POT diffusers transformers accelerate safetensors pyarrow pillow tqdm clean-fid

python -m pytest tests/test_latent_ot.py -q
```

## 2. Download ImageNet-256

If the node cannot access Hugging Face directly, use the mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com python examples/images/imagefolder/download_hf_imagenet256.py \
  --dest ~/datasets/imagenet-1k-256x256 \
  --split train
```

Expected train parquet directory:

```text
~/datasets/imagenet-1k-256x256/data
```

## 3. Encode SD-VAE Latents

Pre-cache the SD-VAE weights once. This is needed because the parallel encoder
uses local-files-only model loading to avoid 8 workers fighting over one model
download/cache lock:

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("cached stabilityai/sd-vae-ft-mse")
PY
```

For 8 H100s, start with:

```bash
cd ~/FlashSinkhorn/FlashWasserstein/conditional-flow-matching-main
source ../env/bin/activate

GPU_LIST=0,1,2,3,4,5,6,7 bash examples/images/latent_imagenet/run_parallel_encode.sh \
  ~/datasets/imagenet-1k-256x256/data \
  ~/datasets/imagenet-1k-256x256/sdvae_latents \
  ~/FlashWasserstein/output/latent_encode_logs \
  8 \
  256 \
  4096
```

The arguments after `run_parallel_encode.sh` are:

```text
DATA_DIR OUT_DIR LOGDIR WORLD_SIZE BATCH_SIZE SHARD_SIZE
```

Monitor:

```bash
tail -f ~/FlashWasserstein/output/latent_encode_logs/encode_r0.log
find ~/datasets/imagenet-1k-256x256/sdvae_latents -name 'latents_*.pt' | wc -l
test -f ~/datasets/imagenet-1k-256x256/sdvae_latents/projection.pt && echo projection_done
```

The encoder builds `projection.pt` after all VAE latent shards finish.

## 4. Coupling Benchmark

Run this before training:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/images/latent_imagenet/bench_latent_ot_coupling.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --context_sizes 2048,4096,8192,16384,32768,65536 \
  --local_batch 256 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --pot_max_context 4096 \
  --dense_max_context 8192 \
  --out_dir ~/FlashWasserstein/output/latent_ot_h100_bench
```

Check that Flash works at `32768` and `65536` contexts before launching long
training.

## 5. Smoke Training

```bash
torchrun --standalone --nproc_per_node=8 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode global_flash_sinkhorn \
  --batch_size 2048 \
  --context_size 32768 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --total_steps 2000 \
  --num_workers 8 \
  --amp \
  --sample_every 1000 \
  --save_step 1000 \
  --output_dir ~/FlashWasserstein/output/latent_train_h100
```

If this is stable, try `--context_size 65536`. If H100 memory is still very
comfortable, try `--batch_size 4096`.

## 6. Main Comparisons

Run one job at a time.

Independent FM:

```bash
torchrun --standalone --nproc_per_node=8 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode independent \
  --batch_size 2048 \
  --total_steps 20000 \
  --num_workers 8 \
  --amp \
  --output_dir ~/FlashWasserstein/output/latent_train_h100
```

Local POT OT-CFM:

```bash
torchrun --standalone --nproc_per_node=8 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode local_pot_exact_row \
  --batch_size 2048 \
  --total_steps 20000 \
  --num_workers 8 \
  --amp \
  --output_dir ~/FlashWasserstein/output/latent_train_h100
```

Flash global OT-CFM:

```bash
torchrun --standalone --nproc_per_node=8 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode global_flash_sinkhorn \
  --batch_size 2048 \
  --context_size 32768 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --total_steps 20000 \
  --num_workers 8 \
  --amp \
  --output_dir ~/FlashWasserstein/output/latent_train_h100
```

Then rerun the Flash job at `--context_size 65536` if the coupling benchmark
looks good.

## 7. Generate Samples

```bash
python examples/images/latent_imagenet/generate_latent_samples.py \
  --checkpoint ~/FlashWasserstein/output/latent_train_h100/<run>/weights_step_00020000.pt \
  --out_dir ~/FlashWasserstein/output/latent_samples_h100/<run> \
  --num_samples 50000 \
  --batch_size 256 \
  --integration_steps 100
```

Use the generated PNG directory for FID/KID tooling.
