# ImageNet Pretraining Guide for TwistNet

## Why Pretraining is Required

TwistNet's STCI module comprises **~65% of total parameters** and is a completely novel design. Unlike SE-ResNet (where SE modules are only ~2% of parameters), borrowing weights from ResNet-18 is insufficient for TwistNet.

| Model | Novel Module Size | Can Borrow from ResNet? |
|-------|-------------------|------------------------|
| SE-ResNet-18 | ~2% | ✅ Yes, works well |
| **TwistNet-18** | **~65%** | ❌ No, needs pretraining |

After pretraining, weights are saved to `weights/twistnet18_imagenet.pt` and will be **automatically detected** by all subsequent experiments.

## Hardware Requirements

| Configuration | 600 epochs (recommended) | 300 epochs (minimum) |
|---------------|-------------------------|---------------------|
| 1x RTX 3090/4090 | ~14 days | ~7 days |
| 4x RTX 3090/4090 | ~4 days | ~2 days |
| 4x A100 | ~3 days | ~1.5 days |
| 8x A100 | ~1.5 days | ~1 day |

## Step-by-Step Guide

### Step 1: Prepare ImageNet Dataset

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 classes)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 classes)
```

### Step 2: Start Pretraining

**Multi-GPU (Recommended)**
```bash
# 4 GPUs - Best quality
torchrun --nproc_per_node=4 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 600 \
    --batch_size 256 \
    --lr 0.1 \
    --checkpoint_dir checkpoints

# 8 GPUs - Faster
torchrun --nproc_per_node=8 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 600 \
    --batch_size 128 \
    --lr 0.1 \
    --checkpoint_dir checkpoints
```

**Single GPU**
```bash
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 600 \
    --batch_size 256 \
    --checkpoint_dir checkpoints
```

**Quick Version (if time is limited)**
```bash
torchrun --nproc_per_node=4 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 300 \
    --checkpoint_dir checkpoints
```

### Step 3: Resume Training (if interrupted)

```bash
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --resume checkpoints/latest.pt
```

### Step 4: After Training Completes

Output files:
```
checkpoints/
├── twistnet18_imagenet.pt   # Final weights
├── best.pt                   # Best validation accuracy checkpoint
├── latest.pt                 # Latest checkpoint (for resume)
└── log.jsonl                 # Training log

weights/
└── twistnet18_imagenet.pt   # ⭐ Auto-detected location
```

### Step 5: Run Downstream Experiments

**No additional parameters needed!** The code auto-detects `weights/twistnet18_imagenet.pt`:

```bash
# Main experiments - automatically uses pretrained weights
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 \
    --run_dir runs/main

# Ablation study - also automatically uses pretrained weights
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 42,43,44 --epochs 100 \
    --run_dir runs/ablation
```

## Auto-Detection Order

The code searches for pretrained weights in this order:

1. `./weights/twistnet18_imagenet.pt` (current working directory)
2. `<script_dir>/weights/twistnet18_imagenet.pt` (script directory)

If not found, it falls back to ResNet-18 partial weights (not recommended, will show warning).

## Expected Results

### ImageNet Validation

| Model | Epochs | Top-1 Acc | Params |
|-------|--------|-----------|--------|
| ResNet-18 (original) | 90 | 69.8% | 11.7M |
| ResNet-18 (timm A1) | 600 | 71.5% | 11.7M |
| **TwistNet-18** | 600 | **72-74%** | 11.6M |
| TwistNet-18 | 300 | 71-73% | 11.6M |

### Downstream Tasks (DTD)

| Method | DTD Acc |
|--------|---------|
| ResNet-18 (ImageNet pretrained) | 68-72% |
| TwistNet-18 (without pretraining) | ~48% ❌ |
| TwistNet-18 (with pretraining) | **73-77%** ✅ |

## Training Configuration

```python
# Recommended settings
{
    "epochs": 600,          # 300 also acceptable
    "batch_size": 256,      # per GPU
    "lr": 0.1,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "randaug": {"n": 2, "m": 9},
}
```

## FAQ

**Q: Can I use fewer epochs?**

A: Yes, 300 epochs is acceptable but may result in 1-2% lower accuracy. For paper submission, 600 epochs is recommended.

**Q: I don't have ImageNet. What are my options?**

A: You can use ImageNet subsets (ImageNet-100, Mini-ImageNet), but results will be suboptimal. Alternatively, contact us for pretrained weights.

**Q: How do I verify pretraining was successful?**

```bash
python train.py --data_dir data/dtd --dataset dtd --fold 1 --seed 42 \
    --model twistnet18 --epochs 10 --run_dir runs/test

# Should show:
# [Auto-detected] Found TwistNet weights: weights/twistnet18_imagenet.pt
# [TwistNet Pretrained] Loaded XXX layers from weights/twistnet18_imagenet.pt
```

**Q: Why 600 epochs?**

A: Modern ImageNet training with strong augmentation (RandAugment, Mixup, CutMix) typically uses 300-600 epochs. The timm library's best ResNet-18 recipe uses 600 epochs to achieve 71.5% top-1 accuracy.
