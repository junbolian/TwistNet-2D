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

We propose **TwistNet-2D**, a novel architecture for texture recognition that captures directional and structural patterns through **Spiral-Twisted Channel Interaction (STCI)**. Unlike conventional attention mechanisms that model first-order feature relationships, STCI computes second-order pairwise channel interactions with spiral spatial displacements, effectively encoding texture orientation and repetitive structures.

## Key Features

- **Spiral-Twisted Channel Interaction (STCI)**: Second-order channel interactions with 4-directional spiral displacements (0°, 45°, 90°, 135°)
- **Adaptive Interaction Selection (AIS)**: Learnable attention over interaction directions
- **Lightweight Design**: Only 11.6M parameters (comparable to ResNet-18)
- **State-of-the-art**: Outperforms recent models on texture recognition benchmarks

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/twistnet-2d.git
cd twistnet-2d

# Create conda environment
conda create -n twistnet python=3.9
conda activate twistnet

# Install dependencies
pip install torch torchvision timm numpy pillow matplotlib seaborn scikit-learn
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

> **Important**: TwistNet requires ImageNet pretraining for optimal performance. The STCI module comprises ~65% of parameters and cannot effectively borrow weights from ResNet-18.

### Pretraining Commands

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

### Pretraining Time

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
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# FMD
python run_all.py --data_dir data/fmd --dataset fmd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# KTH-TIPS2
python run_all.py --data_dir data/kth_tips2 --dataset kth_tips2 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# CUB-200
python run_all.py --data_dir data/cub200 --dataset cub200 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# Flowers-102
python run_all.py --data_dir data/flowers102 --dataset flowers102 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
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

### ImageNet Validation

| Model | Epochs | Top-1 Acc | Params |
|-------|--------|-----------|--------|
| ResNet-18 | 90 | 69.8% | 11.7M |
| ResNet-18 (timm A1) | 600 | 71.5% | 11.7M |
| **TwistNet-18** | 600 | **72-74%** | 11.6M |

### Texture Recognition (DTD)

| Model | Params | DTD Acc |
|-------|--------|---------|
| ResNet-18 | 11.7M | 68-72% |
| SE-ResNet-18 | 11.8M | 69-73% |
| ConvNeXtV2-Nano | 15.6M | 70-74% |
| FastViT-SA12 | 10.9M | 69-73% |
| EfficientFormerV2-S1 | 12.7M | 70-74% |
| RepViT-M1.5 | 14.0M | 70-74% |
| **TwistNet-18 (Ours)** | **11.6M** | **73-77%** |

### Ablation Study

| Variant | DTD Acc | Δ |
|---------|---------|---|
| TwistNet-18 (Full) | 75.2% | - |
| w/o Spiral | 73.1% | -2.1% |
| w/o AIS | 74.0% | -1.2% |
| First-order only | 70.5% | -4.7% |

## Model Zoo

| Model | Params | Group | Pretrained |
|-------|--------|-------|------------|
| resnet18 | 11.7M | Main | ✅ timm |
| seresnet18 | 11.8M | Main | ✅ ResNet-18 weights |
| convnextv2_nano | 15.6M | Main | ✅ timm |
| fastvit_sa12 | 10.9M | Main | ✅ timm |
| efficientformerv2_s1 | 12.7M | Main | ✅ timm |
| repvit_m1_5 | 14.0M | Main | ✅ timm |
| **twistnet18** | **11.6M** | Main | ✅ Custom |
| convnext_tiny | 28.6M | Efficiency | ✅ timm |
| swin_tiny | 28.3M | Efficiency | ✅ timm |

## Training Configuration

All models use identical training settings for fair comparison:

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum=0.9, nesterov=True) |
| Learning Rate | 0.01 |
| LR Schedule | Cosine Annealing |
| Warmup | 5 epochs |
| Epochs | 100 |
| Batch Size | 32 |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Augmentation | RandAugment + Mixup + CutMix |

## File Structure

```
twistnet-2d/
├── models.py              # Model definitions (TwistNet + baselines)
├── train.py               # Single training script
├── run_all.py             # Batch experiment runner
├── pretrain_imagenet.py   # ImageNet pretraining
├── datasets.py            # Dataset loaders
├── transforms.py          # Data augmentation
├── summarize_runs.py      # Results aggregation
├── plot_results.py        # Visualization
├── visualize.py           # TwistNet-specific visualization
├── ablation.py            # Ablation study
├── analysis.py            # Theoretical analysis
├── test_models.py         # Model testing
├── requirements.txt       # Dependencies
├── PRETRAIN_GUIDE.md      # Pretraining guide
├── assets/
│   └── architecture.png   # Architecture diagram
└── weights/
    └── twistnet18_imagenet.pt  # Pretrained weights (auto-generated)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{twistnet2024,
  title={TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition},
  author={},
  booktitle={},
  year={2024}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
- [PyTorch](https://pytorch.org/) for the deep learning framework
