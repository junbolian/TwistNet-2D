"""
Dataset loaders for TwistNet-2D benchmarks.

Supported datasets (5):
1. DTD (Describable Textures Dataset) - 47 classes, 10 official folds
2. FMD (Flickr Material Database) - 10 classes
3. KTH-TIPS2 (KTH Textures under varying Illumination, Pose and Scale) - 11 classes
4. CUB-200-2011 (Caltech-UCSD Birds) - 200 classes
5. Flowers-102 (Oxford 102 Flowers) - 102 classes
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image


# =============================================================================
# Base Dataset Class
# =============================================================================

class ImageListDataset(Dataset):
    """Dataset from a list of (image_path, label) tuples."""
    
    def __init__(self, samples: List[Tuple[str, int]], transform: Callable = None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================================================================
# 1. DTD Dataset (10-fold official splits)
# =============================================================================

class DTDDataset(Dataset):
    """
    Describable Textures Dataset (DTD).
    
    Structure:
        data_dir/
        ├── images/
        │   ├── banded/
        │   └── ...
        └── labels/
            ├── train1.txt, val1.txt, test1.txt
            └── ...
    """
    
    def __init__(self, data_dir: str, split: str = "train", fold: int = 1, transform: Callable = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        images_dir = self.data_dir / "images"
        self.classes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        label_file = self.data_dir / "labels" / f"{split}{fold}.txt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        self.samples = []
        for line in lines:
            cls_name = line.split("/")[0]
            img_path = images_dir / line
            if img_path.exists():
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for DTD {split} fold {fold}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================================================================
# 2. FMD Dataset (Flickr Material Database)
# =============================================================================

def load_fmd(data_dir: str, fold: int = 1, seed: int = 42) -> Tuple[List, List, List, int]:
    """
    FMD: 10 material categories, 100 images each.
    We use 5-fold cross-validation (80/10/10 split per class).
    
    Structure:
        data_dir/
        ├── fabric/
        ├── foliage/
        └── ...
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Use fold-specific seed for reproducibility
    rng = random.Random(seed + fold)
    train_samples, val_samples, test_samples = [], [], []
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        images = sorted([str(p) for p in cls_dir.glob("*.jpg")] + 
                       [str(p) for p in cls_dir.glob("*.png")])
        rng.shuffle(images)
        
        n = len(images)
        fold_size = n // 5
        
        # 5-fold cross-validation
        test_start = (fold - 1) * fold_size
        test_end = test_start + fold_size
        val_start = test_end
        val_end = min(val_start + fold_size, n)
        
        for i, img in enumerate(images):
            label = class_to_idx[cls_name]
            if test_start <= i < test_end:
                test_samples.append((img, label))
            elif val_start <= i < val_end:
                val_samples.append((img, label))
            else:
                train_samples.append((img, label))
    
    return train_samples, val_samples, test_samples, len(classes)


# =============================================================================
# 3. KTH-TIPS2 Dataset
# =============================================================================

def load_kth_tips2(data_dir: str, fold: int = 1, seed: int = 42) -> Tuple[List, List, List, int]:
    """
    KTH-TIPS2-b: 11 material categories, 4 samples × 108 images each.
    Official protocol: Leave-one-sample-out (4-fold), we adapt to 5-fold.
    
    Structure:
        data_dir/
        ├── aluminium_foil/
        │   ├── sample_a/
        │   └── ...
        └── ...
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    rng = random.Random(seed + fold)
    train_samples, val_samples, test_samples = [], [], []
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        samples_dirs = sorted([d for d in cls_dir.iterdir() if d.is_dir()])
        
        # Collect all images
        all_images = []
        for sample_dir in samples_dirs:
            images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
            all_images.extend([str(p) for p in images])
        
        rng.shuffle(all_images)
        n = len(all_images)
        fold_size = n // 5
        
        test_start = (fold - 1) * fold_size
        test_end = test_start + fold_size
        val_end = min(test_end + fold_size, n)
        
        for i, img in enumerate(all_images):
            label = class_to_idx[cls_name]
            if test_start <= i < test_end:
                test_samples.append((img, label))
            elif test_end <= i < val_end:
                val_samples.append((img, label))
            else:
                train_samples.append((img, label))
    
    return train_samples, val_samples, test_samples, len(classes)


# =============================================================================
# 4. CUB-200-2011 Dataset
# =============================================================================

def load_cub200(data_dir: str, fold: int = 1, seed: int = 42) -> Tuple[List, List, List, int]:
    """
    CUB-200-2011: 200 bird species, ~60 images each.
    Official train/test split, we create val from train.
    
    Structure:
        data_dir/
        ├── images/
        │   ├── 001.Black_footed_Albatross/
        │   └── ...
        ├── train_test_split.txt
        └── images.txt
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # Load image list
    with open(data_dir / "images.txt") as f:
        id_to_path = {int(l.split()[0]): l.split()[1] for l in f}
    
    # Load train/test split
    with open(data_dir / "train_test_split.txt") as f:
        id_to_train = {int(l.split()[0]): int(l.split()[1]) for l in f}
    
    # Load class labels
    with open(data_dir / "image_class_labels.txt") as f:
        id_to_label = {int(l.split()[0]): int(l.split()[1]) - 1 for l in f}  # 1-indexed to 0-indexed
    
    rng = random.Random(seed + fold)
    train_images, test_samples = [], []
    
    for img_id, rel_path in id_to_path.items():
        img_path = str(images_dir / rel_path)
        label = id_to_label[img_id]
        if id_to_train[img_id] == 1:
            train_images.append((img_path, label))
        else:
            test_samples.append((img_path, label))
    
    # Split train into train/val (90/10)
    rng.shuffle(train_images)
    val_size = len(train_images) // 10
    val_samples = train_images[:val_size]
    train_samples = train_images[val_size:]
    
    return train_samples, val_samples, test_samples, 200


# =============================================================================
# 5. Flowers-102 Dataset
# =============================================================================

def load_flowers102(data_dir: str, fold: int = 1, seed: int = 42) -> Tuple[List, List, List, int]:
    """
    Oxford Flowers 102: 102 flower categories.
    Official splits: train (1020), val (1020), test (6149).
    
    Structure:
        data_dir/
        ├── jpg/
        │   ├── image_00001.jpg
        │   └── ...
        ├── imagelabels.mat
        └── setid.mat
    """
    import scipy.io as sio
    
    data_dir = Path(data_dir)
    jpg_dir = data_dir / "jpg"
    
    # Load labels (1-indexed)
    labels = sio.loadmat(str(data_dir / "imagelabels.mat"))["labels"][0] - 1
    
    # Load splits
    setid = sio.loadmat(str(data_dir / "setid.mat"))
    train_ids = setid["trnid"][0] - 1  # 1-indexed to 0-indexed
    val_ids = setid["valid"][0] - 1
    test_ids = setid["tstid"][0] - 1
    
    def get_samples(ids):
        samples = []
        for idx in ids:
            img_path = jpg_dir / f"image_{idx+1:05d}.jpg"
            if img_path.exists():
                samples.append((str(img_path), int(labels[idx])))
        return samples
    
    return get_samples(train_ids), get_samples(val_ids), get_samples(test_ids), 102


# =============================================================================
# Unified DataLoader Factory
# =============================================================================

def get_dataloaders(
    data_dir: str,
    dataset: str = "dtd",
    fold: int = 1,
    train_transform: Callable = None,
    eval_transform: Callable = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Get train, val, test dataloaders for any supported dataset.
    
    Args:
        data_dir: Path to dataset root
        dataset: Dataset name (dtd, fmd, kth_tips2, cub200, flowers102)
        fold: Fold number (1-10 for DTD, 1-5 for others)
        train_transform: Transform for training
        eval_transform: Transform for evaluation
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    dataset = dataset.lower().replace("-", "_")
    
    if dataset == "dtd":
        train_ds = DTDDataset(data_dir, "train", fold, train_transform)
        val_ds = DTDDataset(data_dir, "val", fold, eval_transform)
        test_ds = DTDDataset(data_dir, "test", fold, eval_transform)
        num_classes = len(train_ds.classes)
    
    elif dataset == "fmd":
        train_samples, val_samples, test_samples, num_classes = load_fmd(data_dir, fold, seed)
        train_ds = ImageListDataset(train_samples, train_transform)
        val_ds = ImageListDataset(val_samples, eval_transform)
        test_ds = ImageListDataset(test_samples, eval_transform)
    
    elif dataset == "kth_tips2":
        train_samples, val_samples, test_samples, num_classes = load_kth_tips2(data_dir, fold, seed)
        train_ds = ImageListDataset(train_samples, train_transform)
        val_ds = ImageListDataset(val_samples, eval_transform)
        test_ds = ImageListDataset(test_samples, eval_transform)
    
    elif dataset == "cub200":
        train_samples, val_samples, test_samples, num_classes = load_cub200(data_dir, fold, seed)
        train_ds = ImageListDataset(train_samples, train_transform)
        val_ds = ImageListDataset(val_samples, eval_transform)
        test_ds = ImageListDataset(test_samples, eval_transform)
    
    elif dataset == "flowers102":
        train_samples, val_samples, test_samples, num_classes = load_flowers102(data_dir, fold, seed)
        train_ds = ImageListDataset(train_samples, train_transform)
        val_ds = ImageListDataset(val_samples, eval_transform)
        test_ds = ImageListDataset(test_samples, eval_transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: dtd, fmd, kth_tips2, cub200, flowers102")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes


# =============================================================================
# Dataset Info
# =============================================================================

DATASET_INFO = {
    "dtd": {
        "name": "DTD (Describable Textures Dataset)",
        "classes": 47,
        "folds": 10,
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    },
    "fmd": {
        "name": "FMD (Flickr Material Database)",
        "classes": 10,
        "folds": 5,
        "url": "https://people.csail.mit.edu/celiu/CVPR2010/FMD/",
    },
    "kth_tips2": {
        "name": "KTH-TIPS2-b",
        "classes": 11,
        "folds": 5,
        "url": "https://www.csc.kth.se/cvap/databases/kth-tips/",
    },
    "cub200": {
        "name": "CUB-200-2011",
        "classes": 200,
        "folds": 5,
        "url": "http://www.vision.caltech.edu/datasets/cub_200_2011/",
    },
    "flowers102": {
        "name": "Oxford Flowers 102",
        "classes": 102,
        "folds": 5,
        "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    },
}


def print_dataset_info():
    print("=" * 70)
    print("Supported Datasets for TwistNet-2D Benchmark")
    print("=" * 70)
    for key, info in DATASET_INFO.items():
        print(f"\n{info['name']}")
        print(f"  Key: {key}")
        print(f"  Classes: {info['classes']}")
        print(f"  Folds: {info['folds']}")
        print(f"  URL: {info['url']}")
    print("=" * 70)


if __name__ == "__main__":
    print_dataset_info()
