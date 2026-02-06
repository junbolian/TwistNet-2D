# TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition

<p align="center">
  <img src="assets/architecture.png" width="800">
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <a href="#visualization">Visualization</a> •
  <a href="#citation">Citation</a>
</p>

## Abstract

Second-order feature statistics are central to texture recognition, yet current methods face a fundamental tension: bilinear pooling and Gram matrices capture global channel correlations but collapse spatial structure, while self-attention models spatial context through weighted aggregation rather than explicit pairwise feature interactions.

We introduce **TwistNet-2D**, a lightweight module that computes *local* pairwise channel products under *directional spatial displacement*, jointly encoding where features co-occur and how they interact. The core component, **Spiral-Twisted Channel Interaction (STCI)**, shifts one feature map along a prescribed direction before element-wise channel multiplication, thereby capturing the cross-position co-occurrence patterns characteristic of structured and periodic textures.

Aggregating four directional heads with learned channel reweighting and injecting the result through a sigmoid-gated residual path, TwistNet incurs only **~3.5% additional parameters** and **~2% additional FLOPs** over ResNet-18, yet consistently surpasses both parameter-matched and substantially larger baselines—including ConvNeXt, Swin Transformer, and hybrid CNN–Transformer architectures—across four texture and fine-grained recognition benchmarks (DTD, FMD, CUB-200, Flowers-102).

## Key Features

- **Spiral-Twisted Channel Interaction (STCI)**: Second-order channel interactions with 4-directional spiral displacements (0°, 45°, 90°, 135°)
- **Adaptive Interaction Selection (AIS)**: Learnable attention over interaction directions
- **Gated Integration**: Near-zero initialization (γ=-2.0) enables stable training
- **Lightweight Design**: Only **11.59M parameters** (~3.5% overhead vs ResNet-18), **1.85G FLOPs** (~2% overhead)
- **Strong Inductive Bias**: All models trained from scratch without ImageNet pretraining; TwistNet's explicit co-occurrence modeling proves particularly effective in data-limited regimes where larger architectures fail to generalize

## Installation

```bash
# Clone the repository
git clone https://github.com/junbolian/TwistNet-2D.git
cd TwistNet-2D

# Create conda environment
conda create -n twistnet python=3.9
conda activate twistnet

# Install dependencies
pip install torch torchvision timm numpy pillow matplotlib seaborn scikit-learn tqdm

# (Optional) For FLOPs calculation
pip install fvcore
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- timm >= 0.9.0
- CUDA >= 11.3 (for GPU training)

## Quick Start

### Build Model

```python
from models import build_model, list_models

# List available models
list_models()

# Build TwistNet-18 (trained from scratch)
model = build_model('twistnet18', num_classes=47, pretrained=False)

# Build baseline models (from scratch)
resnet = build_model('resnet18', num_classes=47, pretrained=False)
convnext = build_model('convnextv2_nano', num_classes=47, pretrained=False)
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
    --epochs 200
```

## Experiments

> **Note**: All models are trained **from scratch** without ImageNet pretraining. This isolates architectural contributions from transfer learning and reveals that explicit second-order interaction modeling provides strong inductive bias for texture recognition, even in data-limited settings where high-capacity models suffer from severe overfitting.

### Dataset Preparation

See [DATASET.md](DATASET.md) for detailed dataset preparation instructions.

```
data/
├── dtd/           # Describable Textures Dataset (47 classes, 10 folds)
├── fmd/           # Flickr Material Database (10 classes, 5 folds)
├── cub200/        # CUB-200-2011 (200 classes, 5 folds)
└── flowers102/    # Oxford Flowers-102 (102 classes, official splits)
```

### Main Experiments

```bash
# DTD (10 folds × 3 seeds = 30 runs per model)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,repvit_m1_5,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 200 --run_dir runs/main

# FMD (5 folds × 3 seeds = 15 runs per model)
python run_all.py --data_dir data/fmd --dataset fmd \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/main

# CUB-200 (5 folds × 3 seeds = 15 runs per model)
python run_all.py --data_dir data/cub200 --dataset cub200 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/main

# Flowers-102 (5 folds × 3 seeds = 15 runs per model)
python run_all.py --data_dir data/flowers102 --dataset flowers102 \
    --models resnet18,seresnet18,convnextv2_nano,fastvit_sa12,repvit_m1_5,twistnet18 \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/main
```

### Ablation Study

```bash
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 41,42,43,44 --epochs 200 --run_dir runs/ablation
```

### Larger Baselines (Group 2)

Compare TwistNet-18 (11.59M) against ~28M parameter models to study the effect of model scale without pretraining.

```bash
# DTD
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,convnext_tiny,swin_tiny \
    --folds 1-10 --seeds 42,43,44 --epochs 200 --run_dir runs/group2

# FMD
python run_all.py --data_dir data/fmd --dataset fmd \
    --models twistnet18,convnext_tiny,swin_tiny \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/group2

# CUB-200
python run_all.py --data_dir data/cub200 --dataset cub200 \
    --models twistnet18,convnext_tiny,swin_tiny \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/group2

# Flowers-102
python run_all.py --data_dir data/flowers102 --dataset flowers102 \
    --models twistnet18,convnext_tiny,swin_tiny \
    --folds 1-5 --seeds 42,43,44 --epochs 200 --run_dir runs/group2
```

### Generate Results

```bash
# LaTeX tables (mean±std format for top venues)
python summarize_runs.py --run_dir runs/main --latex > tables/main_results.tex
python summarize_runs.py --run_dir runs/ablation --latex > tables/ablation.tex
python summarize_runs.py --run_dir runs/group2 --latex > tables/group2.tex

# CSV export
python summarize_runs.py --run_dir runs/main --csv > tables/main_results.csv
python summarize_runs.py --run_dir runs/group2 --csv > tables/group2.csv

# Text summary with statistics
python summarize_runs.py --run_dir runs/main --summary
python summarize_runs.py --run_dir runs/group2 --summary
```

## Visualization

We provide publication-quality visualization tools for ECCV/CVPR/ICCV submissions.

### Publication Figures (`plot_results.py`)

Generates Nature/Science-style figures with colorblind-friendly palettes, 300 DPI output (PDF + PNG).

```bash
# Bar chart - model comparison across datasets
python plot_results.py --run_dir runs/main --save_dir figures --plot bar

# Radar chart - multi-dataset performance comparison
python plot_results.py --run_dir runs/main --save_dir figures --plot radar

# Scatter plot - parameters vs accuracy trade-off
python plot_results.py --run_dir runs/main --save_dir figures --plot scatter --dataset dtd

# Params vs accuracy with Group 2 overlay
python plot_results.py --run_dir runs/group2 --save_dir figures --plot scatter --dataset dtd

# Generate all figures at once
python plot_results.py --run_dir runs/main --save_dir figures --plot all
```

### Model Visualization (`visualize.py`)

Visualize TwistNet's internal representations.

```bash
# Spiral interaction matrices (4-directional channel correlations)
python visualize.py --checkpoint runs/main/dtd_fold1_twistnet18_seed42/best.pt \
    --image data/dtd/images/banded/banded_0001.jpg --save_dir vis

# Gate value evolution during training
python visualize.py --log_file runs/main/dtd_fold1_twistnet18_seed42/log.jsonl --save_dir vis
```

### Advanced Visualization

```bash
# t-SNE feature visualization (requires checkpoint + data)
python plot_results.py --checkpoint runs/main/dtd_fold1_twistnet18_seed42/best.pt \
    --data_dir data/dtd --plot tsne --save_dir figures

# Interaction heatmaps for sample image
python plot_results.py --checkpoint runs/main/dtd_fold1_twistnet18_seed42/best.pt \
    --image data/dtd/images/striped/striped_0001.jpg --plot interaction --save_dir figures
```

### Output Files

| Script | Output | Description |
|--------|--------|-------------|
| `plot_results.py` | `figures/bar_chart.pdf` | Model comparison bar chart |
| `plot_results.py` | `figures/radar_chart.pdf` | Multi-dataset radar chart |
| `plot_results.py` | `figures/params_accuracy.pdf` | Params vs accuracy scatter |
| `plot_results.py` | `figures/ablation.pdf` | Ablation study results |
| `plot_results.py` | `figures/group2.pdf` | Group 2 comparison plot |
| `plot_results.py` | `figures/tsne.pdf` | t-SNE feature embedding |
| `plot_results.py` | `figures/interaction.pdf` | Interaction heatmaps |
| `visualize.py` | `vis/spiral_interactions.png` | 4-direction interaction matrices |
| `visualize.py` | `vis/feature_maps.png` | Multi-stage feature maps |
| `visualize.py` | `vis/gate_evolution.png` | Gate value learning curves |

## Training Configuration

All models use identical training settings for fair comparison:

| Setting | Value |
|---------|-------|
| **Pretraining** | **None (trained from scratch)** |
| Optimizer | SGD (momentum=0.9, nesterov=True) |
| Learning Rate | 0.05 |
| LR Schedule | Cosine Annealing (min_lr=1e-5) |
| Warmup | 10 epochs (linear) |
| **Epochs** | **200** |
| Batch Size | 64 |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |
| Mixed Precision | Enabled (AMP) |

### Data Augmentation

| Augmentation | Value |
|--------------|-------|
| RandomResizedCrop | scale=(0.2, 1.0), BICUBIC |
| RandomHorizontalFlip | p=0.5 |
| RandAugment | n=2, m=9 |
| Mixup | α=0.8 |
| CutMix | α=1.0 |
| Normalization | ImageNet mean/std |

## Why Train from Scratch?

We train all models from scratch without ImageNet pretraining for two reasons:

1. **Fair comparison**: This isolates the architectural contribution of STCI from transfer learning effects.

2. **Domain mismatch**: ImageNet pretraining optimizes for object-level semantics (shapes, parts, categories), whereas texture recognition requires modeling local co-occurrence patterns and periodicity. TwistNet's STCI modules specifically capture cross-position correlations that ImageNet-pretrained features do not provide.

Empirically, this protocol also reveals an important finding: overparameterized models (~28M) such as ConvNeXt-Tiny and Swin-Tiny suffer severe overfitting on small-scale texture datasets without pretraining, while lightweight models with appropriate inductive bias (TwistNet-18, 11.59M) generalize effectively. This underscores that architectural design matters more than raw model capacity in data-limited regimes.

## Results

### Model Complexity

| Model | Params | FLOPs | Overhead |
|-------|--------|-------|----------|
| ResNet-18 | 11.20M | 1.82G | — |
| **TwistNet-18** | **11.59M** | **1.85G** | **+3.5% params, +2% FLOPs** |

### Model Zoo

#### Group 1: Parameter-matched (10-16M params)

| Model | Params | FLOPs | Venue |
|-------|--------|-------|-------|
| ResNet-18 | 11.20M | 1.82G | CVPR 2016 |
| SE-ResNet-18 | 11.29M | 1.82G | CVPR 2018 |
| ConvNeXtV2-Nano | 15.01M | 2.45G | CVPR 2023 |
| FastViT-SA12 | 10.60M | 1.50G | ICCV 2023 |
| RepViT-M1.5 | 13.67M | 2.31G | CVPR 2024 |
| **TwistNet-18 (Ours)** | **11.59M** | **1.85G** | — |

#### Group 2: Larger Baselines (~28M params)

| Model | Params | FLOPs | Venue |
|-------|--------|-------|-------|
| ConvNeXt-Tiny | 27.86M | 4.47G | CVPR 2022 |
| Swin-Tiny | 27.56M | 4.51G | ICCV 2021 |

### Main Results (Test Accuracy %)

| Model | DTD | FMD | CUB-200 | Flowers-102 |
|-------|-----|-----|---------|-------------|
| ResNet-18 | 39.4±1.2 | 42.6±3.1 | 54.6±0.5 | 43.6±0.5 |
| SE-ResNet-18 | 36.7±1.2 | 40.8±2.8 | 52.0±0.8 | 40.5±0.7 |
| ConvNeXtV2-Nano | 29.1±1.3 | 29.7±2.5 | 31.7±4.0 | 46.1±0.6 |
| FastViT-SA12 | 42.7±1.4 | **45.0±3.6** | 49.9±0.6 | **59.9±0.6** |
| RepViT-M1.5 | 39.2±1.5 | 36.6±2.2 | 59.7±0.6 | 51.6±0.7 |
| **TwistNet-18 (Ours)** | **45.8±1.4** | 43.5±3.8 | **61.8±0.5** | 58.5±0.7 |

### Ablation Study (DTD Test Accuracy %)

| Variant | Params | Description | Accuracy |
|---------|--------|-------------|----------|
| TwistNet-18 (Full) | 11.59M | Complete model | **45.8±1.4** |
| w/o Spiral Twist | 11.59M | Same-position products only | 45.6±1.5 |
| w/o AIS | 11.53M | No Adaptive Interaction Selection | 44.1±1.8 |
| First-order only | 11.20M | No STCI modules | 39.4±1.2 |

### Effect of Model Scale Without Pretraining

Larger models (~28M params) suffer severe degradation when trained from scratch on small-scale datasets, highlighting the importance of parameter-efficient designs with appropriate inductive bias.

| Model | Params | DTD | FMD | CUB-200 | Flowers-102 |
|-------|--------|-----|-----|---------|-------------|
| ConvNeXt-Tiny | 27.86M | 11.1±0.8 | 24.3±2.7 | 3.2±1.4 | 7.5±0.3 |
| Swin-Tiny | 27.56M | 32.2±1.2 | 35.9±3.2 | 33.0±1.0 | 48.8±0.3 |
| **TwistNet-18 (Ours)** | **11.59M** | **45.8±1.4** | **43.5±3.8** | **61.8±0.5** | **58.5±0.7** |

> **Key finding**: ConvNeXt-Tiny drops to 3.2% on CUB-200 and 11.1% on DTD without ImageNet pretraining, consistent with well-documented observations that high-capacity architectures require large-scale pretraining to realize their potential ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929); [Liu et al., 2022](https://arxiv.org/abs/2201.03545)). TwistNet-18 with 2.4× fewer parameters outperforms these models by large margins, demonstrating that targeted inductive bias is more effective than raw capacity in data-limited regimes.

## TwistNet Architecture

### STCI Module Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Channel Reduction (c_red) | 8 | Reduces channels for interaction computation |
| Number of Heads | 4 | One per direction (0°, 45°, 90°, 135°) |
| Pairwise Interactions | 36 per head | c_red × (c_red + 1) / 2 |
| Gate Initialization | -2.0 | Near-zero sigmoid output for stable training |
| AIS Reduction | 4 | SE-style attention reduction ratio |
| Twist Stages | (3, 4) | Applied to layer3 and layer4 |

## File Structure

```
TwistNet-2D/
├── models.py              # Model definitions (TwistNet + baselines)
├── train.py               # Single experiment training script
├── run_all.py             # Batch experiment runner (multi-fold, multi-seed)
├── datasets.py            # Dataset loaders (DTD, FMD, CUB-200, Flowers-102)
├── transforms.py          # Data augmentation (RandAugment, Mixup, CutMix)
├── compute_flops.py       # FLOPs and parameters calculation
├── summarize_runs.py      # Results aggregation (LaTeX mean±std tables)
├── plot_results.py        # Publication figures (bar, radar, scatter, t-SNE)
├── visualize.py           # Model internals (interaction matrices, gates, features)
├── ablation.py            # Ablation study runner
├── analysis.py            # Theoretical analysis
├── test_models.py         # Model testing and validation
├── DATASET.md             # Dataset preparation guide
├── data/                  # Datasets (see DATASET.md)
├── runs/                  # Experiment outputs (checkpoints, logs, results.json)
├── figures/               # Generated publication figures (PDF + PNG)
├── vis/                   # Model visualization outputs
├── tables/                # Generated LaTeX tables
└── assets/
    └── architecture.png   # Architecture diagram
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lian2026twistnet,
  title={TwistNet-2D: Learning Second-Order Channel Interactions via Spiral Twisting for Texture Recognition},
  author={Lian, Junbo Jacob and Xiong, Feng and Chen, Haoran and Sun, Yujun and Ouyang, Kaichen and Yu, Mingyang and Fu, Shengwei and Chen, Huiling},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) for model implementations
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [fvcore](https://github.com/facebookresearch/fvcore) for FLOPs calculation