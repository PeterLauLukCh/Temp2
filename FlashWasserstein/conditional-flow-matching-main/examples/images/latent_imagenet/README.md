# Latent ImageNet OT-CFM Experiments

This folder tests whether large-context OT couplings help latent flow matching
and whether FlashSinkhorn makes those couplings practical.  The unconditional
ImageNet-256 latent path should now be treated as a systems diagnostic, not a
generative-quality benchmark.  For quality claims, use class conditioning.

## 1. Encode ImageNet-256 to SD-VAE Latents

Parallel 10-GPU encode on the A40 node:

```bash
bash examples/images/latent_imagenet/run_parallel_encode.sh
```

To choose specific GPUs, set `GPU_LIST`, for example:

```bash
GPU_LIST=0,1,2,3 bash examples/images/latent_imagenet/run_parallel_encode.sh \
  ~/datasets/imagenet-1k-256x256/data \
  ~/datasets/imagenet-1k-256x256/sdvae_latents \
  ~/FlashWasserstein/output/latent_encode_logs \
  4
```

This writes logs to `~/FlashWasserstein/output/latent_encode_logs`, splits the
40 parquet shards across 10 workers, then builds `projection.pt` after all
workers finish.

Single-process encode:

```bash
python examples/images/latent_imagenet/encode_imagenet256_latents.py \
  --data_dir ~/datasets/imagenet-1k-256x256/data \
  --out_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --batch_size 64 \
  --shard_size 4096 \
  --proj_dim 256 \
  --calibration_samples 65536
```

The encoder saves `latents_*.pt`, `projection.pt`, `metadata.json`, and a
`vae_reconstruction_grid.png` sanity check.

## 2. Coupling Microbenchmark

```bash
python examples/images/latent_imagenet/bench_latent_ot_coupling.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --context_sizes 512,1024,2048,4096,8192,16384 \
  --local_batch 128 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --out_dir outputs/latent_ot_bench
```

Defaults run POT exact only up to context 2048 and dense Sinkhorn up to 4096.
FlashSinkhorn is attempted for every context size.

After the parallel cache finishes, a convenience smoke suite is available:

```bash
bash examples/images/latent_imagenet/run_bench_and_smoke.sh
```

It runs the coupling microbenchmark, 2k-step independent/local-POT smoke runs,
and a shorter Flash global smoke run.

## 3. Smoke Training

Independent FM:

```bash
torchrun --standalone --nproc_per_node=1 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode independent \
  --batch_size 128 \
  --total_steps 2000 \
  --sample_every 1000 \
  --output_dir outputs/latent_train
```

Local OT-CFM baseline:

```bash
torchrun --standalone --nproc_per_node=1 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode local_pot_exact_row \
  --batch_size 128 \
  --total_steps 2000 \
  --output_dir outputs/latent_train
```

Flash global OT-CFM:

```bash
torchrun --standalone --nproc_per_node=10 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode global_flash_sinkhorn \
  --batch_size 1280 \
  --context_size 8192 \
  --eps 0.05 \
  --sinkhorn_iters 80 \
  --total_steps 2000 \
  --output_dir outputs/latent_train
```

Class-conditional latent ImageNet-256:

```bash
torchrun --standalone --nproc_per_node=8 examples/images/latent_imagenet/train_latent_imagenet.py \
  --latent_dir ~/datasets/imagenet-1k-256x256/sdvae_latents \
  --coupling_mode flash_global_entropic \
  --class_conditional \
  --class_aware_coupling \
  --batch_size 2048 \
  --context_size 16384 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  --total_steps 50000 \
  --output_dir outputs/latent_train_classcond
```

The legacy names `global_flash_sinkhorn`, `global_dense_sinkhorn`, and
`local_pot_exact_row` remain accepted aliases.

## 4. Longer Runs

Use the same commands with `--total_steps 20000` first. Extend the best two
methods to `100000` steps only after the smoke metrics and sample grids are
healthy.

## 5. Generate Samples From A Checkpoint

```bash
python examples/images/latent_imagenet/generate_latent_samples.py \
  --checkpoint outputs/latent_train/<run>/weights_step_00020000.pt \
  --out_dir outputs/latent_samples/<run> \
  --num_samples 50000 \
  --batch_size 128 \
  --integration_steps 100
```

For class-conditional checkpoints, add `--class_id <id>` for a single class or
omit it to cycle labels across generated samples.

The generated PNG folder can be passed to external FID/KID tooling. All methods
share the same VAE, latent model, projection, cost normalization, optimizer,
timestep sampler, and sampling code.
