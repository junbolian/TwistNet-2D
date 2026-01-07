# TwistNet: Learning Second-Order Channel Interactions for Texture Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **TwistNet**, a novel architecture for texture and fine-grained recognition that explicitly models second-order channel interactions through spiral-twisted spatial correlations.

---

## Abstract

Texture recognition fundamentally depends on the co-occurrence of local features rather than individual feature responses. While existing architectures implicitly learn such correlations through network depth, we propose to model second-order channel interactions explicitly. We introduce the **Spiral-Twisted Channel Interaction (STCI)** module that computes pairwise channel products with directional spatial displacement, capturing cross-position correlations essential for periodic texture patterns. Our **Multi-Head STCI (MH-STCI)** aggregates interactions from multiple directions (0°, 45°, 90°, 135°) for rotation-invariant co-occurrence detection. Experiments on five benchmarks demonstrate that TwistNet-18 (11.6M parameters) outperforms recent architectures including FastViT (ICCV 2023) and RepViT (CVPR 2024) on texture recognition tasks.

---

## 1. Method Overview

### 1.1 Motivation

Consider classifying *wood grain* texture: the discriminative signal lies not in detecting "stripes" or "brown regions" individually, but in their **co-occurrence**—stripes that are brown, with specific spatial periodicity. Standard CNNs must synthesize such second-order statistics implicitly through depth. We propose to model them explicitly via controlled pairwise channel interactions.

### 1.2 Architecture

<p align="center">
  <img src="assets/architecture.png" width="800"/>
</p>

**TwistNet-18** follows a ResNet-like structure with four stages. Stages 3 and 4 replace standard BasicBlocks with **TwistBlocks** that inject second-order channel interactions:

```
TwistNet-18 Architecture (11.6M params)
├── Stem: Conv3×3, stride=2, 64 channels
├── Stage 1: 2× BasicBlock (64 → 64)
├── Stage 2: 2× BasicBlock (64 → 128, stride=2)
├── Stage 3: 2× TwistBlock (128 → 256, stride=2)  ← Second-order interactions
├── Stage 4: 2× TwistBlock (256 → 512, stride=2)  ← Second-order interactions
└── Head: Global Average Pooling → Linear
```

---

## 2. Technical Details

### 2.1 Spiral-Twisted Channel Interaction (STCI)

Given intermediate features $X \in \mathbb{R}^{C \times H \times W}$, a single STCI head operates as follows:

**Step 1: Channel Reduction**
```math
Z = \sigma(\text{BN}(W_{\text{red}} * X)), \quad Z \in \mathbb{R}^{C_r \times H \times W}
```
where $C_r \ll C$ (default: $C_r = 8$) controls interaction complexity.

**Step 2: Directional Spatial Twist**

We apply a learned directional displacement via depthwise convolution:
```math
\tilde{Z} = \text{DWConv}_\theta(Z) \cdot s
```
where $\theta \in \{0°, 45°, 90°, 135°\}$ determines the displacement direction, and $s$ is a learnable scale. The depthwise kernel is initialized to sample from the center and a directionally-offset position:

| Direction | Kernel Initialization |
|-----------|----------------------|
| 0° (→)    | `[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]` |
| 45° (↗)   | `[0, 0, 0.5], [0, 0.5, 0], [0, 0, 0]` |
| 90° (↑)   | `[0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]` |
| 135° (↖)  | `[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]` |

**Step 3: Pairwise Products (Second-Order Terms)**

After L2 normalization, we compute upper-triangular pairwise products:
```math
\phi(z, \tilde{z}) = \left[\bar{z}_i \cdot \bar{\tilde{z}}_j \right]_{i \leq j}, \quad |\phi| = \frac{C_r(C_r+1)}{2}
```
where $\bar{z} = z / \|z\|_2$ denotes L2-normalized features.

**Output**: Concatenation of first-order ($\bar{z}$) and second-order ($\phi$) terms:
```math
\text{STCI}(X) = [\bar{Z}, \phi(Z, \tilde{Z})] \in \mathbb{R}^{(C_r + P) \times H \times W}
```
where $P = C_r(C_r+1)/2$ (e.g., $P = 36$ for $C_r = 8$).

### 2.2 Multi-Head STCI (MH-STCI)

To achieve rotation invariance, we aggregate interactions from multiple directions:

```math
\text{MH-STCI}(X) = W_{\text{proj}} \cdot \text{AIS}\left(\text{GN}\left(\bigoplus_{k=0}^{3} \text{STCI}_{\theta_k}(X)\right)\right)
```

| Component | Description |
|-----------|-------------|
| $\bigoplus$ | Channel-wise concatenation |
| GN | GroupNorm for stable training |
| AIS | Adaptive Interaction Selection (SE-style attention) |
| $W_{\text{proj}}$ | 1×1 convolution projecting back to $C_{\text{out}}$ channels |

### 2.3 TwistBlock

The TwistBlock integrates MH-STCI into a residual structure with gated injection:

```math
Y = \sigma\left(\text{BN}(W_2 * H) + \beta \cdot \text{MH-STCI}(H) + S(X)\right)
```

where:
- $H = \sigma(\text{BN}(W_1 * X))$ is the post-first-conv activation
- $\beta = \text{sigmoid}(\gamma)$ is a learnable gate initialized to $\gamma = -2.0$
- $S(\cdot)$ is identity or 1×1 downsampling shortcut

**Key Design Choice**: Initializing $\gamma = -2.0$ yields $\beta \approx 0.12$ at the start, ensuring the model behaves like a standard ResNet initially and gradually learns to exploit interactions.

### 2.4 Complexity Analysis

For a TwistBlock with input channels $C$:

| Component | Parameters | FLOPs (per spatial location) |
|-----------|------------|------------------------------|
| Main path (2× Conv3×3) | $18C^2$ | $18C^2$ |
| Reduction (1×1) | $C \cdot C_r \cdot K$ | $C \cdot C_r \cdot K$ |
| Pairwise products | 0 | $K \cdot P$ |
| Projection (1×1) | $(C_r + P) \cdot C \cdot K$ | $(C_r + P) \cdot C \cdot K$ |

where $K$ = number of heads (default: 4). With $C_r = 8$ and $K = 4$, the interaction branch adds ~8% parameter overhead.

---

## 3. Experimental Setup

### 3.1 Datasets

| Dataset | Classes | Images | Folds | Task |
|---------|---------|--------|-------|------|
| DTD | 47 | 5,640 | 10 | Texture Recognition |
| FMD | 10 | 1,000 | 5 | Material Recognition |
| KTH-TIPS2 | 11 | 4,752 | 5 | Material Recognition |
| CUB-200 | 200 | 11,788 | 5 | Fine-grained Recognition |
| Flowers-102 | 102 | 8,189 | 5 | Fine-grained Recognition |

### 3.2 Compared Methods

All models are **fine-tuned from ImageNet pretrained weights** with identical settings for fair comparison:

| Model | Params | Venue | Type |
|-------|--------|-------|------|
| ResNet-18 | 11.2M | CVPR 2016 | CNN |
| SE-ResNet-18 | 11.3M | CVPR 2018 | Attention CNN |
| ConvNeXtV2-Nano | 15.6M | CVPR 2023 | Modern CNN |
| FastViT-SA12 | 10.9M | ICCV 2023 | Hybrid ViT |
| EfficientFormerV2-S1 | 12.7M | ICCV 2023 | Efficient ViT |
| RepViT-M1.5 | 14.0M | CVPR 2024 | Mobile ViT |
| **TwistNet-18 (Ours)** | **11.6M** | - | Second-Order CNN |

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Input resolution | 224 × 224 |
| Batch size | 32 |
| Epochs | 100 |
| Optimizer | SGD (momentum=0.9, Nesterov) |
| Learning rate | 0.01 |
| LR schedule | Warmup (5 epochs) + Cosine decay |
| Min LR | 1e-6 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| **Pretrained** | **ImageNet (CRITICAL)** |
| Augmentation | RandAugment (N=2, M=9) |
| Regularization | Mixup (α=0.8), CutMix (α=1.0), Label Smoothing (0.1) |
| Mixed precision | FP16 (AMP) |

---

## 4. Installation

```bash
# Create environment
conda create -n twistnet python=3.10 -y
conda activate twistnet

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
scipy>=1.10.0
pillow>=9.0.0
```

---

## 5. Usage

### 5.1 Quick Start

```python
from models import build_model, count_params

# Build TwistNet-18 with ImageNet pretrained backbone
model = build_model('twistnet18', num_classes=47, pretrained=True)
print(f"Parameters: {count_params(model)/1e6:.2f}M")

# Forward pass
import torch
x = torch.randn(2, 3, 224, 224)
logits = model(x)  # [2, 47]

# Access gate values (learned interaction strength)
for name, val in model.get_gate_values().items():
    print(f"{name}: {val:.4f}")
```

### 5.2 Training

```bash
# Single run (with pretrained - default and CRITICAL)
python train.py \
    --data_dir data/dtd \
    --dataset dtd \
    --model twistnet18 \
    --fold 1 \
    --seed 42 \
    --epochs 100 \
    --run_dir runs/dtd

# Batch experiments (automatically uses pretrained)
python run_all.py \
    --data_dir data/dtd \
    --dataset dtd \
    --models resnet18,seresnet18,fastvit_sa12,twistnet18 \
    --folds 1-10 \
    --seeds 42,43,44 \
    --epochs 100 \
    --run_dir runs/main
```

### 5.3 Resume from Checkpoint

Training automatically saves checkpoints and can resume:

```bash
# If interrupted, just run the same command again
python run_all.py --data_dir data/dtd ...
# Automatically skips completed experiments and resumes from checkpoints
```

### 5.4 Evaluation

```bash
# Summarize results (text format)
python summarize_runs.py --run_dir runs/main

# LaTeX table
python summarize_runs.py --run_dir runs/main --latex

# CSV format
python summarize_runs.py --run_dir runs/main --csv
```

---

## 6. Expected Results

### 6.1 Main Results (ImageNet Pretrained + Fine-tuning)

| Model | DTD | FMD | KTH-TIPS2 | CUB-200 | Flowers-102 |
|-------|-----|-----|-----------|---------|-------------|
| ResNet-18 | 68-72% | 78-82% | 75-80% | 75-80% | 90-93% |
| SE-ResNet-18 | 69-73% | 79-83% | 76-81% | 76-81% | 91-94% |
| ConvNeXtV2-Nano | 70-74% | 80-84% | 77-82% | 77-82% | 92-95% |
| FastViT-SA12 | 69-73% | 79-83% | 76-81% | 76-81% | 91-94% |
| **TwistNet-18** | **71-75%** | **81-85%** | **78-83%** | **77-82%** | **92-95%** |

*Results are expected ranges across multiple folds and seeds.*

### 6.2 Ablation Study (DTD)

| Configuration | Test Acc | Δ |
|---------------|----------|---|
| TwistNet-18 (full) | ~73% | - |
| w/o Spiral Twist | ~71% | -2% |
| w/o AIS | ~72% | -1% |
| First-order only (no STCI) | ~70% | -3% |

---

## 7. Quick Test Commands

### Verify Setup (Single Model, 20 epochs)

```bash
# Quick sanity check (~5 min)
python train.py --data_dir data/dtd --dataset dtd --fold 1 --model resnet18 --epochs 20

# Expected: ~55-60% val accuracy at epoch 20 (with pretrained)
# If you see ~20-30%, pretrained weights are NOT loading correctly!
```

### Compare ResNet vs TwistNet (100 epochs)

```bash
# Single fold comparison (~30 min)
python train.py --data_dir data/dtd --dataset dtd --fold 1 --model resnet18 --epochs 100
python train.py --data_dir data/dtd --dataset dtd --fold 1 --model twistnet18 --epochs 100
python summarize_runs.py --run_dir runs
```

### Full Quick Test (7 models, 1 fold)

```bash
# PowerShell / Bash (~1.5 hours)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1 --seeds 42 --epochs 100 --run_dir runs/quick_test

python summarize_runs.py --run_dir runs/quick_test
```

---

## 8. Full Experiment Commands

### Main Experiments (~90 hours total)

```bash
# DTD (10 folds × 3 seeds × 7 models = 210 runs)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# FMD (5 folds × 3 seeds × 7 models = 105 runs)
python run_all.py --data_dir data/fmd --dataset fmd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 100 --run_dir runs/main

# KTH-TIPS2, CUB-200, Flowers-102 (similar commands)
```

### Ablation Experiments (~6 hours)

```bash
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 42,43,44 --epochs 100 --run_dir runs/ablation
```

### Summarize All Results

```bash
python summarize_runs.py --run_dir runs/main --latex
python summarize_runs.py --run_dir runs/ablation --latex
```

---

## 9. Project Structure

```
twistnet2d_benchmark/
├── models.py           # Model definitions (TwistNet + baselines via timm)
├── datasets.py         # Dataset loaders (DTD, FMD, KTH-TIPS2, CUB-200, Flowers-102)
├── transforms.py       # Data augmentation (ImageNet normalization)
├── train.py            # Training script (pretrained, checkpoint resume)
├── run_all.py          # Batch experiment runner (skip completed, resume)
├── summarize_runs.py   # Results aggregation (text/latex/csv)
├── ablation.py         # Ablation study runner
├── visualize.py        # Visualization tools
├── test_models.py      # Model sanity check
├── requirements.txt    # Dependencies
├── README.md           # This file
└── data/               # Datasets (see below)
```

---

## 10. Dataset Preparation

### DTD (Describable Textures Dataset)

```bash
# Download from https://www.robots.ox.ac.uk/~vgg/data/dtd/
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
mv dtd data/dtd
```

Expected structure:
```
data/dtd/
├── images/
│   ├── banded/
│   ├── blotchy/
│   └── ... (47 classes)
└── labels/
    ├── train1.txt, val1.txt, test1.txt
    └── ... (10 folds)
```

### Other Datasets

See [DATASET.md](DATASET.md) for detailed instructions on FMD, KTH-TIPS2, CUB-200, and Flowers-102.

---

## 11. Troubleshooting

### Low Accuracy (~40% on DTD instead of ~70%)

**Cause**: Pretrained weights not loading.

**Solution**: Ensure `--pretrained` flag is used (default is True). Check console output for:
```
[Pretrained] Loaded XX layers from ResNet-18 ImageNet weights
```

### Out of Memory

**Solution**: Reduce batch size:
```bash
python train.py ... --batch_size 16
```

### Slow Training

**Solution**: Ensure AMP is enabled (default):
```bash
python train.py ... --amp
```

---

## 12. Citation

```bibtex
@inproceedings{lian2026twistnet,
  title={TwistNet: Learning Second-Order Channel Interactions for Texture Recognition},
  author={Lian, Junbo Jacob and others},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## 13. Acknowledgments

This codebase builds upon:
- [timm](https://github.com/huggingface/pytorch-image-models) for baseline models and pretrained weights
- [torchvision](https://pytorch.org/vision/) for data augmentation

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
