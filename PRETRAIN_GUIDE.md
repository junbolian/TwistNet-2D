# TwistNet ImageNet Pretraining Guide

## 概述

TwistNet 需要专属的 ImageNet 预训练权重，因为 STCI 模块是全新设计的，无法从 ResNet 借用。

**预训练完成后，权重会自动保存到 `weights/twistnet18_imagenet.pt`，后续所有实验（包括消融）都会自动检测并使用。**

## 硬件需求

| 配置 | 600 epochs（推荐） | 300 epochs（最低） |
|------|-------------------|-------------------|
| 1x RTX 3090/4090 | ~14 天 | ~7 天 |
| 4x RTX 3090/4090 | ~4 天 | ~2 天 |
| 4x A100 | ~3 天 | ~1.5 天 |
| 8x A100 | ~1.5 天 | ~1 天 |

**为什么推荐 600 epochs？**
- ResNet-18 在 timm 的最佳配方（A1）使用 600 epochs 达到 71.5%
- TwistNet 的 STCI 模块需要更多迭代来学习二阶交互
- 300 epochs 也可以工作，但可能损失 1-2% 准确率

## 步骤

### Step 1: 准备 ImageNet 数据集

```bash
# ImageNet-1K 目录结构
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 类)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 类)
```

### Step 2: 开始预训练

**单 GPU（约 10-14 天）**
```bash
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 256 \
    --epochs 600 \
    --lr 0.1 \
    --checkpoint_dir checkpoints
```

**多 GPU（推荐，约 3-5 天）**
```bash
# 4 GPU（推荐配置）
torchrun --nproc_per_node=4 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 256 \
    --epochs 600 \
    --lr 0.1 \
    --checkpoint_dir checkpoints
```

**快速验证版（如果时间紧迫）**
```bash
# 300 epochs 也可以，但效果会差 1-2%
torchrun --nproc_per_node=4 pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --epochs 300 \
    --checkpoint_dir checkpoints
```

### Step 3: 断点续训（如果中断）

```bash
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --resume checkpoints/latest.pt
```

### Step 4: 训练完成

训练完成后会自动生成：
```
checkpoints/
├── twistnet18_imagenet.pt   # checkpoint目录的副本
├── best.pt                   # 最佳验证准确率的完整checkpoint
├── latest.pt                 # 最新checkpoint（用于续训）
└── log.jsonl                 # 训练日志

weights/
└── twistnet18_imagenet.pt   # ⭐ 自动检测位置（主要使用）
```

### Step 5: 运行下游实验

**无需任何额外参数！** 代码会自动检测 `weights/twistnet18_imagenet.pt`：

```bash
# 主实验 - 自动使用预训练权重
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 \
    --run_dir runs/main

# 消融实验 - 同样自动使用预训练权重
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18,twistnet18_no_spiral,twistnet18_no_ais,twistnet18_first_order \
    --folds 1-3 --seeds 42,43,44 --epochs 100 \
    --run_dir runs/ablation
```

## 自动检测顺序

代码会按以下顺序自动查找预训练权重：

1. `./weights/twistnet18_imagenet.pt` （当前工作目录）
2. `<script_dir>/weights/twistnet18_imagenet.pt` （脚本所在目录）

如果都找不到，会回退到 ResNet-18 部分权重（不推荐，会有警告）。

## 预期结果

### ImageNet 验证集

| Model | Top-1 Acc | Params |
|-------|-----------|--------|
| ResNet-18 | ~70% | 11.7M |
| TwistNet-18 | **72-74%** | 11.6M |

### 下游任务（DTD）

| 方法 | DTD Acc |
|------|---------|
| ResNet-18 (ImageNet pretrained) | 68-72% |
| TwistNet-18 (无预训练/ResNet部分权重) | ~48% ❌ |
| TwistNet-18 (专属 ImageNet pretrained) | **73-77%** ✅ |

## 常见问题

### Q: 训练时间太长怎么办？

A: 
1. 使用更多 GPU（推荐 4-8 卡）
2. 减少 epochs（200 也可以接受）
3. 使用 ImageNet-100 子集预训练（不推荐，效果会打折扣）

### Q: 没有 ImageNet 数据集怎么办？

A: 可以使用公开的 ImageNet 子集（如 ImageNet-100, ImageNet-Mini），但效果会打折扣。

### Q: 如何验证预训练是否成功？

```bash
# 运行快速测试
python train.py --data_dir data/dtd --dataset dtd --fold 1 --seed 42 \
    --model twistnet18 --epochs 100 --run_dir runs/test

# 查看输出，应该显示：
# [Auto-detected] Found TwistNet weights: weights/twistnet18_imagenet.pt
# [TwistNet Pretrained] Loaded XXX layers from weights/twistnet18_imagenet.pt
```

