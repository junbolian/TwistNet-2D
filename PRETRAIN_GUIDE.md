# ImageNet Pretraining Guide for TwistNet

## Why Pretraining is Required

TwistNet's STCI modules in Stages 3-4 are randomly initialized and require ImageNet pretraining. Standard ResNet-18 weights can only initialize the stem and Stages 1-2 (~35% of parameters), while the TwistBlocks must be trained from scratch.

| Model | Can Use ResNet-18 Weights? | Pretraining Required? |
|-------|---------------------------|----------------------|
| SE-ResNet-18 | ✅ Yes (SE is ~2% params) | ❌ No |
| **TwistNet-18** | ⚠️ Partial (35% params) | ✅ **Yes** |

After pretraining, weights are saved to `weights/twistnet18_imagenet.pt` and will be **automatically detected** by all subsequent experiments.

---

## Step 1: Download ImageNet-1K Dataset

### Dataset Information

| Item | Details |
|------|---------|
| Dataset Name | ImageNet-1K (ILSVRC 2012) |
| Total Size | **~147GB** (this is normal!) |
| Training Images | 1,281,167 images (1000 classes) |
| Validation Images | 50,000 images (1000 classes) |
| Image Format | JPEG |

### Download Options

#### Option A: Official Website (Recommended)

1. **Register an account** at https://image-net.org/download.php
   - Click "Sign up" and fill in your academic email
   - Wait for approval (usually instant for .edu emails)

2. **Download the following files:**
   - `ILSVRC2012_img_train.tar` (~138GB) - Training images
   - `ILSVRC2012_img_val.tar` (~6.3GB) - Validation images
   - `ILSVRC2012_devkit_t12.tar.gz` (~2.5MB) - Development kit (optional)

#### Option B: Kaggle

1. **Create Kaggle account** at https://www.kaggle.com
2. **Join the competition:** https://www.kaggle.com/c/imagenet-object-localization-challenge
3. **Download via Kaggle CLI:**
   ```bash
   pip install kaggle
   kaggle competitions download -c imagenet-object-localization-challenge
   ```

#### Option C: Academic Torrents (Using BitTorrent)

1. **Install a BitTorrent client:**

   **Windows:**
   - Download qBittorrent: https://www.qbittorrent.org/download.php
   - Or use Transmission: https://transmissionbt.com/download/

   **Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt install qbittorrent
   # or
   sudo apt install transmission
   ```

   **macOS:**
   ```bash
   brew install --cask qbittorrent
   # or
   brew install --cask transmission
   ```

2. **Download the torrent file:**
   - Go to: https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2
   - Click "Download torrent" button
   - Save the `.torrent` file

3. **Open with BitTorrent client:**
   - Double-click the `.torrent` file, or
   - In qBittorrent: File → Add Torrent File → Select the `.torrent` file
   - Choose download location (need ~150GB free space)
   - Click "OK" to start downloading

4. **Wait for download:**
   - Speed depends on seeders (usually 10-50 MB/s)
   - Estimated time: 1-4 hours with good connection

---

## Step 2: Extract and Organize Dataset

### Extract Training Data

```bash
# Create directory structure
mkdir -p /path/to/imagenet/{train,val}

# Extract training data (~30 minutes)
cd /path/to/imagenet
tar -xf ILSVRC2012_img_train.tar -C train/

# Extract each class folder (IMPORTANT!)
cd train
for f in *.tar; do
  d="${f%.tar}"
  mkdir -p "$d"
  tar -xf "$f" -C "$d"
  rm "$f"  # Remove tar to save space
done
cd ..
```

### Extract Validation Data

The validation images need to be organized into class folders:

```bash
# Extract validation data
tar -xf ILSVRC2012_img_val.tar -C val/

# Download the organization script
cd val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
cd ..
```

**Alternative: Python script for validation organization:**

```python
import os
import shutil

# Download ILSVRC2012_validation_ground_truth.txt first
# From: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_validation_ground_truth.txt

val_dir = '/path/to/imagenet/val'
ground_truth_file = 'ILSVRC2012_validation_ground_truth.txt'

# Mapping from index to synset (WordNet ID)
# You need to create this mapping from the devkit
synsets = [...]  # List of 1000 synset IDs like 'n01440764'

with open(ground_truth_file) as f:
    labels = [int(line.strip()) - 1 for line in f]  # 1-indexed to 0-indexed

for i, label in enumerate(labels):
    img_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
    src = os.path.join(val_dir, img_name)
    dst_dir = os.path.join(val_dir, synsets[label])
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src, os.path.join(dst_dir, img_name))
```

### Verify Directory Structure

```bash
# Final structure should look like:
/path/to/imagenet/
├── train/
│   ├── n01440764/          # tench (fish)
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ... (~1300 images per class)
│   ├── n01443537/          # goldfish
│   ├── n01484850/          # great white shark
│   └── ... (1000 class folders total)
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ... (50 images per class)
    ├── n01443537/
    └── ... (1000 class folders total)

# Verify counts
find train -name "*.JPEG" | wc -l  # Should be ~1,281,167
find val -name "*.JPEG" | wc -l    # Should be 50,000
ls train | wc -l                    # Should be 1000
ls val | wc -l                      # Should be 1000
```

---

## Step 3: Start Pretraining

### Hardware Requirements

| Configuration | 600 epochs (recommended) | 300 epochs (minimum) |
|---------------|-------------------------|---------------------|
| 1x RTX 3090/4090 | ~14 days | ~7 days |
| 4x RTX 3090/4090 | ~4 days | ~2 days |
| 4x A100 | ~3 days | ~1.5 days |
| 8x A100 | ~1.5 days | ~1 day |

### Training Commands

**Multi-GPU Training (Recommended)**

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

**Single GPU Training**

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

### Resume Training (if interrupted)

```bash
python pretrain_imagenet.py \
    --data_dir /path/to/imagenet \
    --resume checkpoints/latest.pt
```

---

## Step 4: After Training Completes

### Output Files

```
checkpoints/
├── twistnet18_imagenet.pt   # Final weights (use this!)
├── best.pt                   # Best validation accuracy checkpoint
├── latest.pt                 # Latest checkpoint (for resume)
└── log.jsonl                 # Training log

weights/
└── twistnet18_imagenet.pt   # ⭐ Auto-detected location (copy here!)
```

### Copy Weights to Auto-Detection Location

```bash
mkdir -p weights
cp checkpoints/twistnet18_imagenet.pt weights/
```

---

## Step 5: Run Downstream Experiments

**No additional parameters needed!** The code auto-detects `weights/twistnet18_imagenet.pt`:

```bash
# Main experiments - automatically uses pretrained weights
python run_all.py --data_dir data/dtd --dataset dtd \
    --models resnet18,twistnet18 \
    --folds 1-10 --seeds 42,43,44 --epochs 100 \
    --run_dir runs/main

# Verify pretrained weights are loaded:
# You should see:
# [Auto-detected] Found TwistNet weights: weights/twistnet18_imagenet.pt
# [TwistNet Pretrained] Loaded XXX layers from weights/twistnet18_imagenet.pt
```

---

## Training Configuration Details

The pretraining follows the timm A1 recipe for ResNet-18:

```python
{
    "epochs": 600,              # 300 also acceptable
    "batch_size": 256,          # per GPU
    "lr": 0.1,                  # scaled by batch size
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "lr_schedule": "cosine",
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "randaug": {"n": 2, "m": 9},
    "input_size": 224,
}
```

---

## Expected Results

### ImageNet Validation Accuracy

| Model | Epochs | Top-1 Acc | Top-5 Acc |
|-------|--------|-----------|-----------|
| ResNet-18 (original) | 90 | 69.8% | 89.1% |
| ResNet-18 (timm A1) | 600 | 71.5% | 90.4% |
| **TwistNet-18** | 600 | **72-74%** | **90-91%** |
| TwistNet-18 | 300 | 71-73% | 89-90% |

### Downstream Transfer (DTD)

| Method | DTD Accuracy |
|--------|-------------|
| ResNet-18 (ImageNet pretrained) | 68-72% |
| TwistNet-18 (without pretraining) | ~48% ❌ |
| **TwistNet-18 (with pretraining)** | **73-77%** ✅ |

---

## Troubleshooting

### Q: Can I use fewer epochs?

A: Yes, 300 epochs is acceptable but may result in 1-2% lower accuracy. For paper submission, 600 epochs is recommended.

### Q: I don't have ImageNet. What are my options?

A: 
1. Use ImageNet subsets (ImageNet-100, Tiny-ImageNet), but results will be suboptimal
2. Contact us for pretrained weights (after paper acceptance)
3. Use cloud computing (AWS, GCP, Azure) with ImageNet pre-loaded

### Q: How much disk space do I need?

| Item | Size |
|------|------|
| ImageNet download | ~147GB |
| ImageNet extracted | ~147GB |
| Checkpoints | ~500MB |
| **Total** | **~300GB recommended** |

### Q: Training is too slow. How to speed up?

1. Use more GPUs (linear speedup)
2. Enable mixed precision: add `--amp` flag
3. Use faster storage (SSD/NVMe instead of HDD)
4. Increase number of data loading workers: `--workers 8`

### Q: CUDA out of memory error?

Reduce batch size:
```bash
python pretrain_imagenet.py --batch_size 128  # instead of 256
```

### Q: How do I verify pretraining was successful?

```bash
# Quick test on DTD
python train.py --data_dir data/dtd --dataset dtd --fold 1 --seed 42 \
    --model twistnet18 --epochs 10 --run_dir runs/test

# Should show:
# [Auto-detected] Found TwistNet weights: weights/twistnet18_imagenet.pt
# [TwistNet Pretrained] Loaded XXX layers from weights/twistnet18_imagenet.pt
# Epoch 10: val_acc should be > 60% (vs ~30% without pretraining)
```

---

## Auto-Detection Priority

The code searches for pretrained weights in this order:

1. `./weights/twistnet18_imagenet.pt` (current working directory)
2. `<script_dir>/weights/twistnet18_imagenet.pt` (script directory)

If not found, it falls back to ResNet-18 partial weights (not recommended, will show warning).
