# TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **TwistNet-2D**, a novel approach to texture recognition using Spiral-Twisted Channel Interactions.

---

## 🎯 Key Innovation

### Spiral-Twisted Channel Interaction (STCI)

Standard methods compute interactions at the **same spatial position**:
```
interaction(x,y) = z_i(x,y) × z_j(x,y)
```

TwistNet introduces **spiral-twisted interactions** with spatial displacement:
```
interaction(x,y) = z_i(x,y) × z_j(x+δ, y+δ)  
```

where the displacement follows spiral patterns (0°, 45°, 90°, 135°) for **rotation-invariant** co-occurrence detection.

### Why This Matters

- **Texture = Feature Co-occurrence**: Wood grain needs both stripes AND brown variations
- **Cross-position Correlations**: Periodic patterns span multiple pixels  
- **Rotation Invariance**: Multiple spiral directions capture all orientations

---

## 📊 Theoretical Foundation

### 1. Information Theory Perspective
Mutual Information between channels:
```
I(Z_i; Z_j) ≈ -0.5 × log(1 - ρ²_ij)
```
Our pairwise products `z_i × z_j` directly capture the correlation ρ, approximating MI!

### 2. Local vs Global Gram Matrix

| Approach | Formula | Limitation |
|----------|---------|------------|
| Global Gram (Style Transfer) | G_ij = Σ_hw F_ih × F_jh | Loses spatial info |
| **Local Gram (Ours)** | g_ij(x,y) = z_i(x,y) × z_j(x,y) | **Preserves spatial structure** |

---

## 🏆 Benchmark Models

### Group 1: Fair Comparison (10-16M params) - Main Experiments

| Model | Params | Year | Venue | timm name |
|-------|--------|------|-------|-----------|
| ResNet-18 | 11.7M | 2016 | CVPR | `resnet18` |
| SE-ResNet-18 | 11.8M | 2018 | CVPR | `seresnet18` |
| ConvNeXtV2-Nano | 15.6M | 2023 | CVPR | `convnextv2_nano` |
| FastViT-SA12 | 10.9M | 2023 | ICCV | `fastvit_sa12` |
| EfficientFormerV2-S1 | 12.7M | 2023 | ICCV | `efficientformerv2_s1` |
| RepViT-M1.5 | 14.0M | 2024 | CVPR | `repvit_m1_5` |
| **TwistNet-18** | **11.6M** | **Ours** | - | `twistnet18` |

### Group 2: Efficiency Comparison (Official Tiny ~25-30M)

| Model | Params | Year | Venue |
|-------|--------|------|-------|
| ConvNeXt-Tiny | 28.6M | 2022 | CVPR |
| ConvNeXtV2-Tiny | 28.6M | 2023 | CVPR |
| Swin-Tiny | 28.3M | 2021 | ICCV |

---

## 📁 Datasets (5)

| Dataset | Classes | Folds | Description |
|---------|---------|-------|-------------|
| DTD | 47 | 10 | Describable Textures |
| FMD | 10 | 5 | Flickr Material Database |
| KTH-TIPS2 | 11 | 5 | Material Textures |
| CUB-200 | 200 | 5 | Fine-grained Birds |
| Flowers-102 | 102 | 5 | Fine-grained Flowers |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create environment
conda create -n twistnet2d python=3.10 -y
conda activate twistnet2d

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
mkdir -p data && cd data

# DTD (required)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz

# FMD
curl -L -o fmd.zip "https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip"
tar -xf fmd.zip && mv image fmd

# KTH-TIPS2 (download from browser)
# https://www.csc.kth.se/cvap/databases/kth-tips/download.html
# Extract and rename to kth_tips2

# CUB-200
curl -L -o cub200.tar.gz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -xf cub200.tar.gz && mv CUB_200_2011 cub200

# Flowers-102
curl -L -o 102flowers.tgz "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
curl -L -o imagelabels.mat "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
curl -L -o setid.mat "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
mkdir flowers102 && tar -xf 102flowers.tgz -C flowers102
mv imagelabels.mat setid.mat flowers102/
```

### 3. Verify Setup

```bash
# Test all models
python models.py

# Verify all datasets
python -c "from datasets import get_dataloaders; from transforms import build_train_transform, build_eval_transform; t1,t2=build_train_transform(),build_eval_transform(); [print(f'{ds}: OK') for ds in ['dtd','fmd','kth_tips2','cub200','flowers102'] if get_dataloaders(f'data/{ds}',ds,1,t1,t2,batch_size=4,num_workers=0)]"
```

---

## 📋 Complete Experiment Pipeline

### Step 1: Quick Validation (~4 hours)

Test all 7 models on DTD fold 1 with 50 epochs:

```bash
python run_all.py \
    --data_dir data/dtd \
    --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1 \
    --seeds 42 \
    --epochs 50 \
    --run_dir runs/quick_test
```

**Check results before proceeding!** TwistNet should outperform baselines.

---

### Step 2: Main Experiments (~200 GPU hours)

#### 2.1 DTD (47 classes, 10-fold official)

```bash
python run_all.py \
    --data_dir data/dtd \
    --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-10 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/dtd
```
**Runs: 7 models × 10 folds × 3 seeds = 210 runs**

#### 2.2 FMD (10 classes, 5-fold)

```bash
python run_all.py \
    --data_dir data/fmd \
    --dataset fmd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/fmd
```
**Runs: 7 × 5 × 3 = 105 runs**

#### 2.3 KTH-TIPS2 (11 classes, 5-fold)

```bash
python run_all.py \
    --data_dir data/kth_tips2 \
    --dataset kth_tips2 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/kth_tips2
```
**Runs: 7 × 5 × 3 = 105 runs**

#### 2.4 CUB-200 (200 classes, 5-fold)

```bash
python run_all.py \
    --data_dir data/cub200 \
    --dataset cub200 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/cub200
```
**Runs: 7 × 5 × 3 = 105 runs**

#### 2.5 Flowers-102 (102 classes, 5-fold)

```bash
python run_all.py \
    --data_dir data/flowers102 \
    --dataset flowers102 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,efficientformerv2_s1,repvit_m1_5,twistnet18 \
    --folds 1-5 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/flowers102
```
**Runs: 7 × 5 × 3 = 105 runs**

---

### Step 3: Ablation Study (~12 GPU hours)

Test each component's contribution on DTD:

```bash
python run_all.py \
    --data_dir data/dtd \
    --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/ablation
```
**Runs: 4 configs × 3 folds × 3 seeds = 36 runs**

| Model | Description |
|-------|-------------|
| `twistnet18` | Full model |
| `twistnet18_no_spiral` | Without spatial twist |
| `twistnet18_no_ais` | Without adaptive selection |
| `twistnet18_first_order` | Only first-order features |

---

### Step 4: Efficiency Comparison (Optional, ~20 GPU hours)

Compare with larger official models:

```bash
python run_all.py \
    --data_dir data/dtd \
    --dataset dtd \
    --models convnext_tiny,convnextv2_tiny,swin_tiny,twistnet18 \
    --folds 1-5 \
    --seeds 42,43,44 \
    --epochs 200 \
    --run_dir runs/efficiency
```

---

### Step 5: Summarize Results

```bash
# Per-dataset summary
python summarize_runs.py --run_dir runs/dtd --dataset dtd --latex
python summarize_runs.py --run_dir runs/fmd --dataset fmd --latex
python summarize_runs.py --run_dir runs/kth_tips2 --dataset kth_tips2 --latex
python summarize_runs.py --run_dir runs/cub200 --dataset cub200 --latex
python summarize_runs.py --run_dir runs/flowers102 --dataset flowers102 --latex

# Ablation summary
python summarize_runs.py --run_dir runs/ablation --dataset dtd --latex
```

---

## 📊 Experiment Summary

| Experiment | Datasets | Models | Folds | Seeds | Total Runs | Est. Time |
|------------|----------|--------|-------|-------|------------|-----------|
| **Main** | 5 | 7 | 10/5 | 3 | 630 | ~180 hrs |
| **Ablation** | 1 (DTD) | 4 | 3 | 3 | 36 | ~12 hrs |
| **Efficiency** | 1 (DTD) | 4 | 5 | 3 | 60 | ~20 hrs |
| **Total** | - | - | - | - | **726** | **~212 hrs** |

---

## 📈 Visualization & Analysis

### Generate Visualizations

```bash
# Gram matrices & feature maps
python visualize.py \
    --checkpoint runs/dtd/dtd_fold1_twistnet18_seed42/best.pt \
    --image data/dtd/images/banded/banded_0001.jpg \
    --save_dir vis/

# Gate evolution during training
python visualize.py \
    --log_file runs/dtd/dtd_fold1_twistnet18_seed42/log.jsonl \
    --save_dir vis/
```

### Theoretical Analysis

```bash
python analysis.py \
    --data_dir data/dtd \
    --dataset dtd \
    --checkpoint runs/dtd/dtd_fold1_twistnet18_seed42/best.pt \
    --analysis all \
    --save_dir analysis/
```

Outputs:
- `mi_analysis.png` - Mutual information between channels
- `gram_analysis.png` - Local vs global Gram comparison
- `class_patterns.png` - Class-specific co-occurrence patterns
- `theoretical_report.md` - Theoretical foundation document

---

## 📁 Project Structure

```
twistnet2d_benchmark/
├── models.py              # All models (timm + TwistNet)
├── datasets.py            # Dataset loaders (5 datasets)
├── transforms.py          # Data augmentation
├── train.py               # Training script
├── run_all.py             # Batch experiment runner
├── summarize_runs.py      # Results aggregation
├── ablation.py            # Ablation study runner
├── visualize.py           # Visualization tools
├── analysis.py            # Theoretical analysis
├── requirements.txt       # Dependencies
├── README.md              # This file
└── data/
    ├── dtd/
    ├── fmd/
    ├── kth_tips2/
    ├── cub200/
    └── flowers102/
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 |
| Batch size | 64 |
| Epochs | 200 |
| Optimizer | SGD + Nesterov |
| Learning rate | 0.05 |
| LR schedule | Warmup(10) + Cosine |
| Weight decay | 1e-4 |
| Augmentation | RandAugment(N=2,M=9) + Mixup(0.8) + CutMix(1.0) |
| Label smoothing | 0.1 |

---

## 🔧 Model Architecture

```
TwistNet-18 Architecture:
├── Stem: Conv3×3 (stride 2)
├── Stage 1: 2× BasicBlock (64ch)
├── Stage 2: 2× BasicBlock (128ch)
├── Stage 3: 2× TwistBlock (256ch)  ← Spiral-Twisted Interaction
├── Stage 4: 2× TwistBlock (512ch)  ← Spiral-Twisted Interaction
└── Head: GAP → FC

TwistBlock:
├── Conv3×3 → BN → ReLU → Conv3×3 → BN
├── Multi-Head Spiral-Twisted Interaction (MH-STCI)
│   ├── Head 0: Direction 0° (→)
│   ├── Head 1: Direction 45° (↗)
│   ├── Head 2: Direction 90° (↑)
│   └── Head 3: Direction 135° (↖)
├── Adaptive Interaction Selection (AIS)
└── Gated residual connection
```

---

## 📖 Citation

```bibtex
@inproceedings{twistnet2025,
  title={TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition},
  author={Junbo Jacob Lian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## 📚 References

- ResNet: He et al., "Deep Residual Learning", CVPR 2016
- SE-Net: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
- ConvNeXtV2: Woo et al., "ConvNeXt V2", CVPR 2023
- FastViT: Vasu et al., "FastViT", ICCV 2023
- EfficientFormerV2: Li et al., "EfficientFormerV2", ICCV 2023
- RepViT: Wang et al., "RepViT", CVPR 2024

---

## 📝 License

This project is licensed under the MIT License.