# Flash Global OT-CFM Protocol

This protocol replaces the earlier unconditional ImageNet-256 latent run as
the main experimental path.

The theoretical argument is recorded in
[`FLASH_GLOBAL_ENTROPIC_OTCFM_THEORY.md`](FLASH_GLOBAL_ENTROPIC_OTCFM_THEORY.md).
The short version is: CFM only requires a valid endpoint coupling; entropic OT
is valid and has controlled cost bias; FlashSinkhorn makes large-context
entropic OT practical; and global entropic OT can beat exact local POT whenever
the global-vs-local context gain exceeds the entropic bias.

## Stage 1: CIFAR-10 Pixel-Space

Run the coupling microbenchmark first:

```bash
python examples/images/cifar10/bench_ot_coupling.py \
  --context_sizes 512,1024,2048,4096,8192,16384 \
  --local_batch 128 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  --out_dir outputs/cifar10_ot_bench
```

Then run short training sweeps:

```bash
torchrun --standalone --nproc_per_node=1 examples/images/cifar10/train_cifar10_global_ot.py \
  --coupling_mode independent \
  --batch_size 128 \
  --total_steps 50000 \
  --sample_every 5000 \
  --output_dir outputs/cifar10_global_ot

torchrun --standalone --nproc_per_node=1 examples/images/cifar10/train_cifar10_global_ot.py \
  --coupling_mode local_exact_pot \
  --batch_size 128 \
  --total_steps 50000 \
  --sample_every 5000 \
  --output_dir outputs/cifar10_global_ot

torchrun --standalone --nproc_per_node=8 examples/images/cifar10/train_cifar10_global_ot.py \
  --coupling_mode flash_global_entropic \
  --batch_size 1024 \
  --context_size 8192 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  --total_steps 50000 \
  --sample_every 5000 \
  --output_dir outputs/cifar10_global_ot
```

Only launch 400k-step final runs after these sample grids are non-noise.

Evaluate a completed CIFAR sweep directly from checkpoints:

```bash
python examples/images/cifar10/evaluate_cifar10_global_ot.py \
  --run_root outputs/cifar10_global_ot \
  --step 50000 \
  --num_gen 50000 \
  --integration_method euler \
  --integration_steps 100 \
  --out_json outputs/cifar10_global_ot/fid_step50000.json
```

For a quick but noisy read, use `--num_gen 10000` first.

## Stage 2: ImageNet-64 Pixel-Space

Use an ImageFolder-format ImageNet-64 train root.

```bash
torchrun --standalone --nproc_per_node=8 examples/images/imagefolder/train_imagefolder_global_ot.py \
  --data_root /path/to/imagenet64/train \
  --coupling_mode flash_global_entropic \
  --batch_size 800 \
  --context_size 8192 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  --image_size 64 \
  --total_steps 250000 \
  --output_dir outputs/imagenet64_global_ot
```

Run the same command with `independent`, `local_exact_pot`, and
`local_entropic` before treating Flash as a result.

For Parquet ImageNet diagnostics, summarize the theory condition after running
the global-vs-local feature OT benchmark:

```bash
python examples/images/imagefolder/summarize_ot_theory_condition.py \
  --root ~/FlashSinkhorn/output/imagenet64_phase1_feature_ot \
  --eps_values 0.01,0.02,0.05 \
  --out_json ~/FlashSinkhorn/output/imagenet64_phase1_feature_ot/theory_condition.json
```

## Stage 3: ImageNet-256 Latent

Use this only after CIFAR-10 or ImageNet-64 shows a real signal.  The quality
benchmark should be class-conditional:

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

## Required Evidence

- `metrics.jsonl` must show `ot_time_s`, `step_time_s`, `images_per_s`,
  `sample_cost`, and `duplicate_fraction`.
- Flash global OT must be compared against local exact POT and local entropic
  OT, not only independent FM.
- Coupling diagnostics should report block-local exact cost, global exact cost,
  entropic sampled cost, cross-block fraction, and whether
  `W_block - W_global > eps * log(n)` after the same cost normalization.
- The claim is promising only if larger context improves FID, low-NFE FID,
  path straightness, or time-to-target-FID at matched backbone and data.
