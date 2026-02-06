# Dataset Preparation Guide

This guide explains how to download and organize the four benchmark datasets for TwistNet-2D experiments.

## Overview

| Dataset | Classes | Images | Folds | Task | Download Size |
|---------|---------|--------|-------|------|---------------|
| DTD | 47 | 5,640 | 10 | Texture Recognition | ~600 MB |
| FMD | 10 | 1,000 | 5 | Material Recognition | ~250 MB |
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

> **5-Fold Cross-Validation Protocol**
>
> FMD uses standard 5-fold cross-validation with 60/20/20 split (train/val/test).
> Each fold rotates which portion is used for testing, with validation wrapping around:
> - Fold 1: test=0, val=1, train=2,3,4
> - Fold 2: test=1, val=2, train=0,3,4
> - Fold 3: test=2, val=3, train=0,1,4
> - Fold 4: test=3, val=4, train=0,1,2
> - Fold 5: test=4, val=0, train=1,2,3

```bash
# Download from official website
# https://people.csail.mit.edu/celiu/CVPR2010/FMD/

# Extract to data/fmd/
unzip FMD.zip -d data/fmd/
```

**Download Link:** [FMD Dataset](https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip)

### 3. CUB-200-2011 (Caltech-UCSD Birds)

```bash
# Download
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# Extract
tar -xzf CUB_200_2011.tgz -C data/

# Rename
mv data/CUB_200_2011 data/cub200
```

**Download Link:** [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

### 4. Flowers-102 (Oxford Flowers)

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
datasets = ['dtd', 'fmd', 'cub200', 'flowers102']
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
[OK] cub200: ~12000 files
[OK] flowers102: ~8200 files
```

## Quick Start After Setup

```bash
# Test with DTD dataset (10 folds)
python run_all.py --data_dir data/dtd --dataset dtd \
    --models twistnet18 --folds 1 --seeds 42 --epochs 10
```

## Troubleshooting

### "Dataset not found" error
- Check that the folder names match exactly: `dtd`, `fmd`, `cub200`, `flowers102`
- Ensure images are not nested in extra folders

### "scipy not found" for Flowers-102
```bash
pip install scipy
```

### Large file extraction issues
- Use 7-Zip for better compatibility on Windows
- Ensure you have enough disk space (~3 GB total)

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