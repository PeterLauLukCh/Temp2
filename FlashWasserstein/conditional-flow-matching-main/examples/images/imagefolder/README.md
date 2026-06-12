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

## Single-GPU Training

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

## 10-GPU Training

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
