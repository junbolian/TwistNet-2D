# TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition

## 概述

TwistNet-2D 是一种用于纹理识别的新型网络架构，核心创新是 **Spiral-Twisted Channel Interaction (STCI)** 模块，通过螺旋位移和通道间二阶交互来捕获纹理的方向性和空间结构信息。

## 文件结构

```
twistnet_code_v2/
├── models.py              # TwistNet + Baseline 模型定义
├── train.py               # 单次训练脚本
├── run_all.py             # 批量实验运行器
├── pretrain_imagenet.py   # ImageNet 预训练脚本
├── datasets.py            # 数据集加载（DTD, FMD, KTH, CUB, Flowers）
├── transforms.py          # 数据增强
├── summarize_runs.py      # 结果汇总 & LaTeX 表格生成
├── plot_results.py        # 可视化图表生成
├── visualize.py           # TwistNet 特有可视化
├── ablation.py            # 消融实验
├── analysis.py            # 理论分析工具
├── test_models.py         # 模型测试
├── requirements.txt       # 依赖
├── PRETRAIN_GUIDE.md      # 预训练详细指南
└── weights/               # 预训练权重存放位置
    └── twistnet18_imagenet.pt  # (预训练后自动生成)
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision timm numpy pillow matplotlib seaborn
```

### 2. 准备数据集

```
data/
├── dtd/           # Describable Textures Dataset
├── fmd/           # Flickr Material Database
├── kth_tips2/     # KTH-TIPS2
├── cub200/        # CUB-200-2011
└── flowers102/    # Oxford Flowers-102
```

### 3. 完整实验流程

#### Phase 1: ImageNet 预训练（必须）

TwistNet 需要专属预训练权重，因为 STCI 模块是全新设计。

```bash
# 单 GPU（约14天）
python pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 600

# 多 GPU（推荐，约3-4天）
torchrun --nproc_per_node=4 pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 600

# 快速版（时间紧迫时，约2天，效果略差）
torchrun --nproc_per_node=4 pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 300
```

**预训练完成后**，权重会自动保存到 `weights/twistnet18_imagenet.pt`，后续实验会自动检测并加载。

#### Phase 2: 运行实验

**预训练完成后，直接运行以下命令即可（无需额外参数）：**

```bash
# ============================================================
# 主实验（7个模型 × 5数据集）
# ============================================================

# DTD
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

# ============================================================
# 消融实验（只在 DTD 上做）
# ============================================================
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 42,43,44 --epochs 100 --run_dir runs/ablation
```

#### Phase 3: 生成结果

```bash
# ============================================================
# 1. 生成 LaTeX 表格
# ============================================================
python summarize_runs.py --run_dir runs/main --latex > tables/main_results.tex
python summarize_runs.py --run_dir runs/ablation --latex > tables/ablation.tex

# ============================================================
# 2. 生成可视化图表
# ============================================================

# 主实验柱状图
python plot_results.py --run_dir runs/main --save_dir figures --plot bar

# 消融实验图
python plot_results.py --run_dir runs/ablation --save_dir figures --plot ablation

# Gate 值演化（展示 STCI 学习过程）
python plot_results.py \
    --log_file runs/main/dtd_fold1_twistnet18_seed42/log.jsonl \
    --save_dir figures \
    --plot gate

# 交互矩阵热图（放 Method 部分）
python plot_results.py \
    --checkpoint runs/main/dtd_fold1_twistnet18_seed42/best.pt \
    --image data/dtd/images/banded/banded_0001.jpg \
    --save_dir figures \
    --plot interaction
```

## 训练设置

所有模型使用统一的训练设置以保证公平比较：

| 参数 | 值 |
|------|-----|
| Optimizer | SGD (momentum=0.9, nesterov=True) |
| Learning Rate | 0.01 |
| LR Schedule | Cosine Annealing |
| Warmup | 5 epochs |
| Epochs | 100 |
| Batch Size | 32 |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Augmentation | RandAugment + Mixup + CutMix |
| Pretrained | ImageNet (所有模型) |

## 模型列表

### Group 1: 公平对比（10-16M 参数）

| 模型 | 参数量 | 来源 |
|------|--------|------|
| ResNet-18 | 11.7M | CVPR 2016 |
| SE-ResNet-18 | 11.8M | CVPR 2018 |
| ConvNeXtV2-Nano | 15.6M | CVPR 2023 |
| FastViT-SA12 | 10.9M | ICCV 2023 |
| EfficientFormerV2-S1 | 12.7M | ICCV 2023 |
| RepViT-M1.5 | 14.0M | CVPR 2024 |
| **TwistNet-18 (Ours)** | **11.6M** | - |

### Group 2: 效率对比（25-30M 参数）

| 模型 | 参数量 | 来源 |
|------|--------|------|
| ConvNeXt-Tiny | 28.6M | CVPR 2022 |
| Swin-Tiny | 28.3M | ICCV 2021 |
| MaxViT-Tiny | 30.9M | ECCV 2022 |

## 预训练权重自动检测

代码会按以下顺序自动检测 TwistNet 预训练权重：

1. `./weights/twistnet18_imagenet.pt` （当前目录）
2. `<script_dir>/weights/twistnet18_imagenet.pt` （脚本目录）

如果找不到，会回退到 ResNet-18 部分权重（不推荐）。

## 预期结果

### ImageNet 验证集

| 模型 | Epochs | Top-1 Acc | 参数量 |
|------|--------|-----------|--------|
| ResNet-18 (原始) | 90 | ~69.8% | 11.7M |
| ResNet-18 (timm A1) | 600 | ~71.5% | 11.7M |
| **TwistNet-18** | 600 | **72-74%** | 11.6M |
| **TwistNet-18** | 300 | **71-73%** | 11.6M |

### DTD 数据集

| 模型 | Accuracy |
|------|----------|
| ResNet-18 | 68-72% |
| ConvNeXtV2-Nano | 70-74% |
| **TwistNet-18** | **73-77%** |

## 可视化输出

| 图表 | 文件 | 用途 |
|------|------|------|
| 主实验柱状图 | `figures/bar_chart.pdf` | Results section |
| 消融实验图 | `figures/ablation.pdf` | Ablation study |
| Gate 值演化 | `figures/gate_evolution.pdf` | Analysis |
| 交互矩阵热图 | `figures/interaction.pdf` | Method section |

## 常见问题

### Q: 为什么 TwistNet 必须预训练？

A: TwistNet 的 STCI 模块是全新设计，无法从 ResNet 借用权重。如果不预训练，STCI 模块（占模型约 65%）需要从头学习，在小数据集上效果很差。

### Q: 预训练需要多长时间？

| GPU 配置 | 600 epochs（推荐） | 300 epochs（快速） |
|----------|-------------------|-------------------|
| 1x RTX 3090 | ~14 天 | ~7 天 |
| 4x RTX 3090 | ~4 天 | ~2 天 |
| 4x A100 | ~3 天 | ~1.5 天 |

### Q: 如何断点续训？

```bash
# ImageNet 预训练
python pretrain_imagenet.py --data_dir /path/to/imagenet --resume checkpoints/latest.pt

# 下游任务
python run_all.py --data_dir data/dtd --dataset dtd --models twistnet18 ...
# (自动跳过已完成的实验，从 checkpoint 恢复未完成的)
```

### Q: 如何查看训练进度？

```bash
# 查看已完成的实验
python summarize_runs.py --run_dir runs/main

# 查看具体某次实验的日志
cat runs/main/dtd_fold1_twistnet18_seed42/log.jsonl
```

## 引用

如果您使用了本代码，请引用：

```bibtex
@article{twistnet2024,
  title={TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
