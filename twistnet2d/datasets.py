"""
Dataset loaders for TwistNet-2D benchmarks.

Datasets:
1. DTD (47 classes, 10-fold official)
2. FMD (10 classes, 5-fold)
3. KTH-TIPS2 (11 classes, 5-fold)
4. CUB-200 (200 classes, 5-fold)
5. Flowers-102 (102 classes, 5-fold)
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ImageListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform: Callable = None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DTDDataset(Dataset):
    """DTD with official 10-fold splits."""
    def __init__(self, data_dir: str, split: str = "train", fold: int = 1, transform: Callable = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        images_dir = self.data_dir / "images"
        self.classes = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        label_file = self.data_dir / "labels" / f"{split}{fold}.txt"
        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        self.samples = []
        for line in lines:
            cls_name = line.split("/")[0]
            img_path = images_dir / line
            if img_path.exists():
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_fmd(data_dir: str, fold: int = 1, seed: int = 42):
    """Load FMD dataset with 5-fold split."""
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    random.seed(seed)
    train_samples, val_samples, test_samples = [], [], []
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        images = sorted([str(p) for p in cls_dir.glob("*.jpg")])
        random.shuffle(images)
        n = len(images)
        fold_size = n // 5
        test_start = (fold - 1) * fold_size
        test_end = test_start + fold_size
        val_end = min(test_end + fold_size, n)
        
        for i, img in enumerate(images):
            label = class_to_idx[cls_name]
            if test_start <= i < test_end:
                test_samples.append((img, label))
            elif test_end <= i < val_end:
                val_samples.append((img, label))
            else:
                train_samples.append((img, label))
    
    return train_samples, val_samples, test_samples, len(classes)


def load_kth_tips2(data_dir: str, fold: int = 1, seed: int = 42):
    """Load KTH-TIPS2 dataset."""
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    random.seed(seed)
    train_samples, val_samples, test_samples = [], [], []
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        all_images = []
        for sample_dir in cls_dir.iterdir():
            if sample_dir.is_dir():
                all_images.extend([str(p) for p in sample_dir.glob("*.png")])
                all_images.extend([str(p) for p in sample_dir.glob("*.jpg")])
        
        random.shuffle(all_images)
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


def load_cub200(data_dir: str, fold: int = 1, seed: int = 42):
    """Load CUB-200-2011 dataset."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    with open(data_dir / "images.txt") as f:
        id_to_path = {int(l.split()[0]): l.split()[1] for l in f}
    with open(data_dir / "train_test_split.txt") as f:
        id_to_train = {int(l.split()[0]): int(l.split()[1]) for l in f}
    with open(data_dir / "image_class_labels.txt") as f:
        id_to_label = {int(l.split()[0]): int(l.split()[1]) - 1 for l in f}
    
    random.seed(seed + fold)
    train_images, test_samples = [], []
    
    for img_id, rel_path in id_to_path.items():
        img_path = str(images_dir / rel_path)
        label = id_to_label[img_id]
        if id_to_train[img_id] == 1:
            train_images.append((img_path, label))
        else:
            test_samples.append((img_path, label))
    
    random.shuffle(train_images)
    val_size = len(train_images) // 10
    val_samples = train_images[:val_size]
    train_samples = train_images[val_size:]
    
    return train_samples, val_samples, test_samples, 200


def load_flowers102(data_dir: str, fold: int = 1, seed: int = 42):
    """Load Oxford Flowers-102 dataset."""
    import scipy.io as sio
    
    data_dir = Path(data_dir)
    jpg_dir = data_dir / "jpg"
    
    labels = sio.loadmat(str(data_dir / "imagelabels.mat"))["labels"][0] - 1
    setid = sio.loadmat(str(data_dir / "setid.mat"))
    train_ids = setid["trnid"][0] - 1
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


def get_dataloaders(
    data_dir: str,
    dataset: str = "dtd",
    fold: int = 1,
    train_transform: Callable = None,
    eval_transform: Callable = None,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
):
    """Get train, val, test dataloaders."""
    dataset = dataset.lower().replace("-", "_")
    
    if dataset == "dtd":
        train_ds = DTDDataset(data_dir, "train", fold, train_transform)
        val_ds = DTDDataset(data_dir, "val", fold, eval_transform)
        test_ds = DTDDataset(data_dir, "test", fold, eval_transform)
        num_classes = len(train_ds.classes)
    elif dataset == "fmd":
        train_s, val_s, test_s, num_classes = load_fmd(data_dir, fold, seed)
        train_ds = ImageListDataset(train_s, train_transform)
        val_ds = ImageListDataset(val_s, eval_transform)
        test_ds = ImageListDataset(test_s, eval_transform)
    elif dataset == "kth_tips2":
        train_s, val_s, test_s, num_classes = load_kth_tips2(data_dir, fold, seed)
        train_ds = ImageListDataset(train_s, train_transform)
        val_ds = ImageListDataset(val_s, eval_transform)
        test_ds = ImageListDataset(test_s, eval_transform)
    elif dataset == "cub200":
        train_s, val_s, test_s, num_classes = load_cub200(data_dir, fold, seed)
        train_ds = ImageListDataset(train_s, train_transform)
        val_ds = ImageListDataset(val_s, eval_transform)
        test_ds = ImageListDataset(test_s, eval_transform)
    elif dataset == "flowers102":
        train_s, val_s, test_s, num_classes = load_flowers102(data_dir, fold, seed)
        train_ds = ImageListDataset(train_s, train_transform)
        val_ds = ImageListDataset(val_s, eval_transform)
        test_ds = ImageListDataset(test_s, eval_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes


DATASET_INFO = {
    "dtd": {"name": "DTD", "classes": 47, "folds": 10},
    "fmd": {"name": "FMD", "classes": 10, "folds": 5},
    "kth_tips2": {"name": "KTH-TIPS2", "classes": 11, "folds": 5},
    "cub200": {"name": "CUB-200", "classes": 200, "folds": 5},
    "flowers102": {"name": "Flowers-102", "classes": 102, "folds": 5},
}