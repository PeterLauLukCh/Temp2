# ImageFolder Flash OT-CFM

This folder provides large-dataset image entry points for FlashWasserstein inside
OT-CFM. The dataset must follow torchvision `ImageFolder` layout, for example:

```text
/data/imagenet/train/
  class_a/*.JPEG
  class_b/*.JPEG
```

## Download ImageNet-1K

ImageNet-1K / ILSVRC2012 is gated. Create a Kaggle account, accept the
ImageNet Object Localization Challenge rules, then create an API token and put
`kaggle.json` at `~/.kaggle/kaggle.json`.

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

./download_imagenet_kaggle.sh ~/datasets/imagenet
```

The script prints the final `--data_root`, usually one of:

```text
~/datasets/imagenet/extracted/ILSVRC/Data/CLS-LOC/train
~/datasets/imagenet/train
```

## Download Tiny ImageNet

Tiny ImageNet is a small public ImageNet-derived benchmark with 200 classes and
64x64 images. It is useful for quick OT-CFM smoke tests, but it is not a
replacement for ImageNet-1K-scale evidence.

```bash
./download_tiny_imagenet.sh ~/datasets/tiny-imagenet
```

Use the printed train root as `--data_root`, typically:

```text
~/datasets/tiny-imagenet/tiny-imagenet-200/train
```

## Pairing Benchmark

Run this before full training:

```bash
CUDA_VISIBLE_DEVICES=0 python bench_ot_pairing.py \
  --data_root /data/imagenet/train \
  --image_size 64 \
  --batch_sizes 128,256,512,1024 \
  --methods exact,flash \
  --out ./ot_pairing_bench_imagenet64
```

Use `--max_pot_batch` to skip POT at large batches.

## Standard ImageNet-64 / ImageFolder Global OT-CFM

Use `train_imagefolder_global_ot.py` for the standardized pixel-space
ImageNet-64 benchmark. It supports `independent`, `local_exact_pot`,
`local_entropic`, `allgather_dense_entropic`, `global_pot_exact_small`, and
`flash_global_entropic`.

The default model settings in this script are the ImageNet-64 pilot defaults:
channels `192`, residual blocks `3`, channel multipliers `1,2,3,4`, attention
at resolution `8`, dropout `0.1`, and LR `1e-4`.

```bash
torchrun --standalone --nproc_per_node=8 train_imagefolder_global_ot.py \
  --data_root /path/to/imagenet64/train \
  --coupling_mode independent \
  --batch_size 800 \
  --image_size 64 \
  --total_steps 250000 \
  --output_dir ./runs_imagenet64

torchrun --standalone --nproc_per_node=8 train_imagefolder_global_ot.py \
  --data_root /path/to/imagenet64/train \
  --coupling_mode local_exact_pot \
  --batch_size 800 \
  --image_size 64 \
  --total_steps 250000 \
  --output_dir ./runs_imagenet64

torchrun --standalone --nproc_per_node=8 train_imagefolder_global_ot.py \
  --data_root /path/to/imagenet64/train \
  --coupling_mode flash_global_entropic \
  --batch_size 800 \
  --context_size 8192 \
  --eps 0.05 \
  --sinkhorn_iters 20 \
  --image_size 64 \
  --total_steps 250000 \
  --output_dir ./runs_imagenet64
```

## HF Parquet ImageNet-64 Global OT-CFM

For local Hugging Face Parquet shards such as
`benjamin-paine/imagenet-1k-256x256`, use
`train_hf_parquet_global_ot.py` directly instead of converting the dataset to
ImageFolder:

```bash
DATA_DIR=~/datasets/imagenet-1k-256x256/data
OUT=~/FlashSinkhorn/output/imagenet64_parquet_smoke_20k
COMMON="--data_dir $DATA_DIR --image_size 64 --batch_size 1024 --total_steps 20000 --num_workers 4 --arrow_batch_size 256 --amp --class_conditional --sample_every 5000 --save_step 10000 --log_step 20 --output_dir $OUT --cost_feature_dim 256 --lr 1e-4"

torchrun --standalone --nproc_per_node=8 train_hf_parquet_global_ot.py \
  $COMMON \
  --coupling_mode local_exact_pot

torchrun --standalone --nproc_per_node=8 train_hf_parquet_global_ot.py \
  $COMMON \
  --coupling_mode flash_global_entropic \
  --context_size 32768 \
  --eps 0.01 \
  --sinkhorn_iters 30
```

The first parquet experiments should keep class-conditional generation enabled
but leave `--class_aware_coupling` off.  Class-aware OT is an ablation because
ImageNet-1K has too many classes for small local minibatches.

## Legacy Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_imagefolder.py \
  --data_root /data/imagenet/train \
  --model otcfm \
  --ot_method flash \
  --image_size 64 \
  --batch_size 256 \
  --total_steps 100000 \
  --output_dir ./runs
```

## Legacy 10-GPU Training

```bash
torchrun --standalone --nproc_per_node=10 train_imagefolder.py \
  --data_root /data/imagenet/train \
  --model otcfm \
  --ot_method flash \
  --image_size 64 \
  --batch_size 1280 \
  --total_steps 100000 \
  --output_dir ./runs
```

In DDP mode, `--batch_size` is the global batch size and must be divisible by
the number of ranks. OT pairing is currently local to each GPU's minibatch. This
matches the usual per-device minibatch training pattern, but it is not a single
global cross-GPU OT solve.
