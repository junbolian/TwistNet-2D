# TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition

<p align="center">
  <img src="assets/architecture.png" width="800">
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pretrain-on-imagenet">Pretrain</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

## Abstract

Second-order feature statistics are fundamental to texture recognition, yet existing approaches face a trade-off: bilinear pooling and Gram matrices capture global correlations but discard spatial structure, while self-attention models spatial dependencies through weighted sums rather than explicit feature interactions.

We propose **TwistNet-2D**, a lightweight module that computes *local* pairwise channel products with *directional spatial displacement*, preserving both where and how features co-occur. Our **Spiral-Twisted Channel Interaction (STCI)** shifts one feature map along a specified direction before computing channel-wise products, capturing the cross-position correlations characteristic of periodic textures.

By aggregating four directional heads with adaptive channel weighting and injecting via a sigmoid-gated residual connection, TwistNet adds only **~3.5% parameters** and **~2% FLOPs** to ResNet-18 while consistently outperforming ResNet, SE-ResNet, ConvNeXt, and hybrid CNN-transformer baselines on five texture and fine-grained benchmarks (DTD, FMD, KTH-TIPS2, CUB-200, Flowers-102).

## Key Features

- **Spiral-Twisted Channel Interaction (STCI)**: Second-order channel interactions with 4-directional spiral displacements (0°, 45°, 90°, 135°)
- **Adaptive Interaction Selection (AIS)**: Learnable attention over interaction directions, selecting which co-occurrence patterns matter for each image
- **Gated Integration**: Near-zero initialization (γ=-2.0) enables stable insertion into pretrained CNNs
- **Lightweight Design**: Only **11.59M parameters** (~3.5% overhead vs ResNet-18), **1.85G FLOPs** (~2% overhead)

## Installation

```bash
# Clone the repository
git clone https://github.com/junbolian/TwistNet-2D.git
cd TwistNet-2D

# Create conda environment
conda create -n twistnet python=3.9
conda activate twistnet

# Install dependencies
pip install torch torchvision timm numpy pillow matplotlib seaborn scikit-learn

# (Optional) For FLOPs calculation
pip install fvcore
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- timm >= 0.9.0
- CUDA >= 11.3 (for GPU training)

## Quick Start

### Build Model

```python
from models import build_model, list_models

# List available models
list_models()

# Build TwistNet-18 with pretrained weights
model = build_model('twistnet18', num_classes=47, pretrained=True)

# Build baseline models
resnet = build_model('resnet18', num_classes=47, pretrained=True)
convnext = build_model('convnextv2_nano', num_classes=47, pretrained=True)
```

### Compute FLOPs and Parameters

```bash
# All models
python compute_flops.py

# Single model
python compute_flops.py --model twistnet18

# LaTeX table output
python compute_flops.py --latex
```

### Single Training Run

```bash
python train.py \
    --data_dir data/dtd \
    --dataset dtd \
    --model twistnet18 \
    --fold 1 \
    --seed 42 \
    --epochs 100 \
    --pretrained
```

## Pretrain on ImageNet

> **Important**: TwistNet requires ImageNet pretraining for optimal performance. Standard ResNet-18 weights only cover ~35% of TwistNet's architecture (stem + layer1 + layer2), while the STCI modules in layer3/layer4 require training from scratch.

### Step 1: Download ImageNet-1K Dataset

You must manually download ImageNet-1K (ILSVRC 2012) dataset (~150GB):

| Source | Link | Notes |
|--------|------|-------|
| **Official** | https://image-net.org/download.php | Requires registration (free for academic use) |
| Kaggle | https://www.kaggle.com/c/imagenet-object-localization-challenge | Requires Kaggle account |
| Academic Torrents | https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2 | Torrent download |

After downloading, organize the dataset as follows:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ... (1000 class folders)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 class folders)
```

### Step 2: Run Pretraining

```bash
# Multi-GPU training (Recommended: 4x A100, ~3-4 days)
torchrun --nproc_per_node=4 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 600 \
    --batch_size 256 \
    --lr 0.1 \
    --checkpoint_dir checkpoints

# Single GPU (slower, ~14 days)
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 600

# Resume from checkpoint
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --resume checkpoints/latest.pt
```

### Pretraining Time Estimates

| Configuration | 600 epochs | 300 epochs |
|--------------|------------|------------|
| 1x RTX 3090 | ~14 days | ~7 days |
| 4x RTX 3090 | ~4 days | ~2 days |
| 4x A100 | ~3 days | ~1.5 days |

After pretraining, weights are automatically saved to `weights/twistnet18_imagenet.pt` and will be auto-detected by `build_model()`.

## Experiments

### Dataset Preparation

```
data/
├── dtd/           # Describable Textures Dataset
├── fmd/           # Flickr Material Database  
├── kth_tips2/     # KTH-TIPS2
├── cub200/        # CUB-200-2011
└── flowers102/    # Oxford Flowers-102
```

### Main Experiments

```bash
# DTD (10 folds × 3 seeds = 30 runs per model)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s2,repvit_m1_5,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# FMD
python run_all.py --data_dir data/fmd --dataset fmd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s2,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# KTH-TIPS2
python run_all.py --data_dir data/kth_tips2 --dataset kth_tips2 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s2,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# CUB-200
python run_all.py --data_dir data/cub200 --dataset cub200 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s2,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# Flowers-102
python run_all.py --data_dir data/flowers102 --dataset flowers102 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s2,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main
```

### Ablation Study

```bash
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 42,43,44 --epochs 100 --run_dir runs/ablation
```

### Generate Results

```bash
# LaTeX tables
python summarize_runs.py --run_dir runs/main --latex > tables/main_results.tex
python summarize_runs.py --run_dir runs/ablation --latex > tables/ablation.tex

# Figures
python plot_results.py --run_dir runs/main --save_dir figures --plot bar
python plot_results.py --run_dir runs/ablation --save_dir figures --plot ablation
python plot_results.py --log_file runs/main/dtd_fold1_twistnet18_seed42/log.jsonl --save_dir figures --plot gate
python plot_results.py --checkpoint runs/main/dtd_fold1_twistnet18_seed42/best.pt \
    --image data/dtd/images/banded/banded_0001.jpg --save_dir figures --plot interaction
```

## Results

### Parameter and FLOPs Overhead

| Model | Params | FLOPs | Overhead |
|-------|--------|-------|----------|
| ResNet-18 | 11.20M | 1.82G | — |
| **TwistNet-18** | **11.59M** | **1.85G** | **+3.5% params, +2% FLOPs** |

### Model Zoo

#### Group 1: Fair Comparison (10–16M params)

| Model | Params | FLOPs | Venue | Pretrained |
|-------|--------|-------|-------|------------|
| ResNet-18 | 11.20M | 1.82G | CVPR 2016 | ✅ timm |
| SE-ResNet-18 | 11.29M | 1.82G | CVPR 2018 | ✅ timm |
| ConvNeXtV2-Nano | 15.01M | 2.45G | CVPR 2023 | ✅ timm |
| FastViT-SA12 | 10.60M | 1.50G | ICCV 2023 | ✅ timm |
| EfficientFormerV2-S2 | 12.70M | 1.30G | ICCV 2023 | ✅ timm |
| RepViT-M1.5 | 13.67M | 2.31G | CVPR 2024 | ✅ timm |
| **TwistNet-18 (Ours)** | **11.59M** | **1.85G** | — | ✅ Custom |

#### Group 2: Larger Baselines (~28M params)

| Model | Params | FLOPs | Venue | Pretrained |
|-------|--------|-------|-------|------------|
| ConvNeXt-Tiny | 27.86M | 4.47G | CVPR 2022 | ✅ timm |
| Swin-Tiny | 27.56M | 4.51G | ICCV 2021 | ✅ timm |

### Ablation Variants

| Variant | Params | Description |
|---------|--------|-------------|
| TwistNet-18 (Full) | 11.59M | Complete model with all components |
| w/o Spiral Twist | 11.59M | Same-position products only (no directional displacement) |
| w/o AIS | 11.53M | No Adaptive Interaction Selection |
| First-order only | 11.20M | No STCI modules (equivalent to ResNet-18 backbone) |

## Training Configuration

All models use identical training settings for fair comparison:

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum=0.9, nesterov=True) |
| Learning Rate | 0.01 |
| LR Schedule | Cosine Annealing (min_lr=1e-6) |
| Warmup | 5 epochs (linear) |
| Epochs | 100 |
| Batch Size | 32 |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Augmentation | RandAugment (n=2, m=9) + Mixup (α=0.8) + CutMix (α=1.0) |

## File Structure

```
TwistNet-2D/
├── models.py              # Model definitions (TwistNet + baselines)
├── train.py               # Single training script
├── run_all.py             # Batch experiment runner
├── pretrain_imagenet.py   # ImageNet pretraining
├── compute_flops.py       # FLOPs and parameters calculation
├── datasets.py            # Dataset loaders
├── transforms.py          # Data augmentation
├── summarize_runs.py      # Results aggregation
├── plot_results.py        # Visualization
├── visualize.py           # TwistNet-specific visualization
├── ablation.py            # Ablation study
├── analysis.py            # Theoretical analysis
├── test_models.py         # Model testing
├── requirements.txt       # Dependencies
├── PRETRAIN_GUIDE.md      # Detailed pretraining guide
├── assets/
│   └── architecture.png   # Architecture diagram
└── weights/
    └── twistnet18_imagenet.pt  # Pretrained weights (auto-generated after pretraining)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lian2025twistnet,
  title={TwistNet-2D: Learning Second-Order Channel Interactions via Spiral Twisting for Texture Recognition},
  author={Lian, Junbo Jacob and Xiong, Feng and Chen, Haoran and Sun, Yujun and Ouyang, Kaichen and Yu, Mingyang and Fu, Shengwei and Chen, Huiling},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models and training recipes
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [fvcore](https://github.com/facebookresearch/fvcore) for FLOPs calculation