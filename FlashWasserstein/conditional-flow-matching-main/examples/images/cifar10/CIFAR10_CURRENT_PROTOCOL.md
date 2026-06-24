# CIFAR-10 Current FlashSinkhorn Protocol

This note records the current clean CIFAR-10 experimental protocol after the
June 2026 debugging pass. It supersedes the exploratory projected-cost and
row-conditional POT runs for the main CIFAR claim.

## Training Setup

- Dataset: CIFAR-10, pixel space, images normalized to `[-1, 1]`.
- Model: TorchCFM CIFAR U-Net.
- Optimizer: Adam, learning rate `2e-4`.
- Warmup: scale with image budget.
  - Batch 128 / 400k steps: warmup `5000`.
  - Batch 256 / 200k steps: warmup `2500`.
- EMA: `0.9999`.
- Dropout: `0.1`.
- Gradient clip: `1.0`.
- AMP: enabled.
- Validation loss: disabled for final comparison. It is method-specific and is
  not a fair cross-method metric.
- Primary metric: CleanFID from 50k generated samples, CIFAR-10 train reference
  stats, `legacy_tensorflow`, EMA checkpoint, Euler sampler.

## Main Fair Comparison

Use two GPUs per training job, global batch 256, local batch 128 per GPU, and
200k steps. This gives the same total image budget as batch 128 for 400k steps:

```text
200000 * 256 = 51.2M images
400000 * 128 = 51.2M images
```

The official OT-CFM baseline solves exact OT independently on each GPU over the
local batch, so with global batch 256 and two GPUs it still solves `128 x 128`
exact minibatch OT. FlashSinkhorn all-gathers across the two GPUs and uses a
large target queue, so it evaluates a global large-context coupling.

Use full-pixel cost for both methods:

```bash
--cost_feature_dim 0
```

With the normalized cost scale in `OTCouplingSampler`, CIFAR-10 full-pixel
features have dimension `3 * 32 * 32 = 3072`, and the effective cost scale is
`1 / (2 * 3072)`. Therefore CLI eps values such as `0.02` and `0.03` are
normalized eps values. Do not multiply these by the pixel dimension unless the
code is changed to raw unscaled cost.

## Clean Result Table

Configuration: CIFAR-10, full-pixel cost, two GPUs per job, global batch 256,
local batch 128 per GPU, 200k steps, 51.2M images seen, 50k generated samples
against CIFAR-10 train CleanFID stats.

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

The train-reference 50k FID is the paper-standard CleanFID protocol. The
test-reference 10k result is a corroborative held-out check, but it has larger
sampling noise.

## Interpretation

The current main claim should be:

```text
At matched per-GPU batch size 128 and matched total images seen, FlashSinkhorn
with an 8K full-pixel context improves CIFAR-10 FID over the official exact
OT-CFM baseline across Euler NFE values.
```

Avoid saying the optimizer batch is matched when comparing the two-GPU batch
256 runs with older single-GPU batch 128 runs. The correct wording is:

```text
same per-GPU batch size and same total images seen
```

## What Not To Use As Main Evidence

- `local_exact_pot` in `train_cifar10_global_ot.py` is not the official OT-CFM
  baseline. It is a custom row-conditional POT coupling.
- Projected-cost runs with `--cost_feature_dim 256` are useful ablations, but
  they are not the full-pixel official-cost comparison.
- Method-specific train loss and validation loss are not fair cross-method
  metrics because each method induces different target velocity pairs.
- Exact POT at large contexts such as `8192 x 8192` is not practical inside
  training; this is part of the motivation for FlashSinkhorn.

## Current Tuning Direction

The promising region is full-pixel FlashSinkhorn with batch 128 per GPU,
contexts around 2K-16K, and eps near `0.02-0.03`. Initial results suggest 8K is
stronger than 2K/4K and 12K in the tested settings, but more tuning is needed.

