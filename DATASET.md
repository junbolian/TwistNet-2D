# Dataset Preparation Guide

This guide explains how to download and organize the five benchmark datasets for TwistNet-2D experiments.

## Overview

| Dataset | Classes | Images | Folds | Task | Download Size |
|---------|---------|--------|-------|------|---------------|
| DTD | 47 | 5,640 | 10 | Texture Recognition | ~600 MB |
| FMD | 10 | 1,000 | 5 | Material Recognition | ~250 MB |
| KTH-TIPS2 | 11 | 4,752 | 4 (LOSO) | Material Recognition | ~1.8 GB |
| CUB-200-2011 | 200 | 11,788 | 5 | Fine-grained Recognition | ~1.2 GB |
| Flowers-102 | 102 | 8,189 | official | Fine-grained Recognition | ~330 MB |

## Directory Structure

After setup, your `data/` folder should look like:

```
data/
├── dtd/
│   ├── images/
│   │   ├── banded/
│   │   ├── blotchy/
│   │   └── ... (47 classes)
│   └── labels/
│       ├── train1.txt ... train10.txt
│       ├── val1.txt ... val10.txt
│       └── test1.txt ... test10.txt
├── fmd/
│   ├── fabric/
│   ├── foliage/
│   └── ... (10 classes)
├── kth_tips2/
│   ├── aluminium_foil/
│   │   ├── sample_a/
│   │   ├── sample_b/
│   │   ├── sample_c/
│   │   └── sample_d/
│   ├── brown_bread/
│   └── ... (11 classes)
├── cub200/
│   ├── images/
│   │   ├── 001.Black_footed_Albatross/
│   │   ├── 002.Laysan_Albatross/
│   │   └── ... (200 classes)
│   ├── images.txt
│   ├── image_class_labels.txt
│   └── train_test_split.txt
└── flowers102/
    ├── jpg/
    │   ├── image_00001.jpg
    │   └── ... (8189 images)
    ├── imagelabels.mat
    └── setid.mat
```

## Download Instructions

### 1. DTD (Describable Textures Dataset)

```bash
# Download
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

# Extract
tar -xzf dtd-r1.0.1.tar.gz -C data/
```

**Download Link:** [DTD Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)

### 2. FMD (Flickr Material Database)

```bash
# Download from official website
# https://people.csail.mit.edu/celiu/CVPR2010/FMD/

# Extract to data/fmd/
unzip FMD.zip -d data/fmd/
```

**Download Link:** [FMD Dataset](https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip)

### 3. KTH-TIPS2

> **IMPORTANT: Leave-One-Sample-Out (LOSO) Protocol**
>
> KTH-TIPS2 requires special handling to prevent data leakage. Each material class contains 4 physical samples (a, b, c, d), with each sample photographed under multiple lighting conditions and scales.
>
> **Why LOSO matters:** If images are randomly split, different lighting/scale variants of the same physical sample will appear in both training and test sets. The model can then "memorize" sample appearance rather than learning material properties, leading to artificially inflated accuracy (often 100% on validation/test).
>
> **Our implementation:**
> - Fold 1: Train on samples a, b, c → Test on sample d
> - Fold 2: Train on samples a, b, d → Test on sample c
> - Fold 3: Train on samples a, c, d → Test on sample b
> - Fold 4: Train on samples b, c, d → Test on sample a

```bash
# Download
wget https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b.tar

# Extract
tar -xf kth-tips2-b.tar -C data/

# Rename
mv data/KTH-TIPS2-b data/kth_tips2
```

**Download Link:** [KTH-TIPS2-b](https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b.tar)

**Expected structure after extraction:**
```
data/kth_tips2/
├── aluminium_foil/
│   ├── sample_a/
│   │   ├── 01-scale_1_im_1_col.png
│   │   └── ... (36+ images)
│   ├── sample_b/
│   ├── sample_c/
│   └── sample_d/
├── brown_bread/
│   ├── sample_a/
│   ├── sample_b/
│   ├── sample_c/
│   └── sample_d/
└── ... (11 material classes total)
```

### 4. CUB-200-2011 (Caltech-UCSD Birds)

```bash
# Download
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# Extract
tar -xzf CUB_200_2011.tgz -C data/

# Rename
mv data/CUB_200_2011 data/cub200
```

**Download Link:** [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

### 5. Flowers-102 (Oxford Flowers)

```bash
# Create directory
mkdir -p data/flowers102

# Download images
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# Download labels
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# Download splits
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

# Extract images
tar -xzf 102flowers.tgz -C data/flowers102/

# Move label files
mv imagelabels.mat data/flowers102/
mv setid.mat data/flowers102/
```

**Download Links:**
- [Images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
- [Labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)
- [Splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat)

## Windows Users

If you're on Windows without `wget`, you can:

1. **Use PowerShell:**
```powershell
Invoke-WebRequest -Uri "URL" -OutFile "filename"
```

2. **Use browser:** Click the download links directly

3. **Use 7-Zip:** Extract `.tar.gz` and `.tgz` files

## Verification

Run this command to verify your setup:

```bash
python -c "
import os
datasets = ['dtd', 'fmd', 'kth_tips2', 'cub200', 'flowers102']
for d in datasets:
    path = f'data/{d}'
    if os.path.exists(path):
        n_files = sum(len(files) for _, _, files in os.walk(path))
        print(f'[OK] {d}: {n_files} files')
    else:
        print(f'[MISSING] {d}: NOT FOUND')
"
```

Expected output:
```
[OK] dtd: ~5700 files
[OK] fmd: ~1000 files
[OK] kth_tips2: ~4800 files
[OK] cub200: ~12000 files
[OK] flowers102: ~8200 files
```

## Quick Start After Setup

```bash
# Test with DTD dataset (10 folds)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18 --folds 1 --seeds 42 --epochs 10

# Test with KTH-TIPS2 dataset (4 folds LOSO)
python run_all.py --data_dir data/kth_tips2 --dataset kth_tips2 \
    --models twistnet18 --folds 1-4 --seeds 42 --epochs 10
```

## Troubleshooting

### "Dataset not found" error
- Check that the folder names match exactly: `dtd`, `fmd`, `kth_tips2`, `cub200`, `flowers102`
- Ensure images are not nested in extra folders

### KTH-TIPS2 shows 100% validation accuracy
- This indicates data leakage from incorrect splitting
- Make sure you're using our `datasets.py` which implements LOSO protocol
- Do NOT use random train/test splits for KTH-TIPS2

### "scipy not found" for Flowers-102
```bash
pip install scipy
```

### Large file extraction issues
- Use 7-Zip for better compatibility on Windows
- Ensure you have enough disk space (~5 GB total)

## Citation

If you use these datasets, please cite the original papers:

```bibtex
@inproceedings{cimpoi2014dtd,
  title={Describing textures in the wild},
  author={Cimpoi, M. and Maji, S. and Kokkinos, I. and Mohamed, S. and Vedaldi, A.},
  booktitle={CVPR},
  year={2014}
}

@inproceedings{sharan2009fmd,
  title={Material perception: What can you see in a brief glance?},
  author={Sharan, L. and Rosenholtz, R. and Adelson, E.},
  booktitle={Journal of Vision},
  year={2009}
}

@article{caputo2005kthtips,
  title={Class-specific material categorisation},
  author={Caputo, B. and Hayman, E. and Mallikarjuna, P.},
  journal={ICCV},
  year={2005}
}

@techreport{wah2011cub200,
  title={The Caltech-UCSD Birds-200-2011 Dataset},
  author={Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
  institution={California Institute of Technology},
  year={2011}
}

@inproceedings{nilsback2008flowers102,
  title={Automated flower classification over a large number of classes},
  author={Nilsback, M-E. and Zisserman, A.},
  booktitle={ICVGIP},
  year={2008}
}
```
