#!/usr/bin/env python3
"""
pretrain_imagenet.py - Pretrain TwistNet on ImageNet-1K

This script trains TwistNet from scratch on ImageNet to create
proper pretrained weights where STCI modules are fully trained.

Requirements:
- ImageNet-1K dataset (~150GB)
- 4-8 GPUs recommended (or 1 GPU with gradient accumulation)
- ~3-5 days training time (4 GPU, 600 epochs)

Recommended epochs:
- 600 epochs: Best quality (recommended)
- 300 epochs: Acceptable quality, faster

Usage:
    # Single GPU (slow, ~14 days for 600 epochs)
    python pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 600
    
    # Multi-GPU with DDP (recommended, ~3-4 days on 4x A100)
    torchrun --nproc_per_node=4 pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 600
    
    # Quick version (300 epochs, ~2 days on 4x A100)
    torchrun --nproc_per_node=4 pretrain_imagenet.py --data_dir /path/to/imagenet --epochs 300
    
    # Resume from checkpoint
    python pretrain_imagenet.py --data_dir /path/to/imagenet --resume checkpoints/latest.pt
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from models import TwistNet, count_params


# =============================================================================
# Configuration
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_args():
    parser = argparse.ArgumentParser(description="Pretrain TwistNet on ImageNet")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageNet (with train/val folders)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Model
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--twist_stages", type=str, default="3,4", help="Stages with STCI")
    parser.add_argument("--gate_init", type=float, default=-2.0)
    
    # Training
    parser.add_argument("--epochs", type=int, default=600,
                        help="Training epochs (300=minimum, 600=recommended for best results)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # Augmentation
    parser.add_argument("--randaug_n", type=int, default=2)
    parser.add_argument("--randaug_m", type=int, default=9)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    
    # System
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()


# =============================================================================
# Data
# =============================================================================

def build_train_transform(img_size, randaug_n, randaug_m):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=randaug_n, magnitude=randaug_m),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform(img_size):
    return T.Compose([
        T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataloaders(args, rank, world_size):
    train_transform = build_train_transform(args.img_size, args.randaug_n, args.randaug_m)
    val_transform = build_val_transform(args.img_size)
    
    train_dataset = ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(args.data_dir, "val"), transform=val_transform)
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


# =============================================================================
# Mixup / CutMix
# =============================================================================

def mixup_cutmix(x, y, mixup_alpha, cutmix_alpha, num_classes=1000):
    """Apply mixup or cutmix with 50% probability each."""
    if random.random() > 0.5 and mixup_alpha > 0:
        # Mixup
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        indices = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[indices]
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        y_onehot_perm = y_onehot[indices]
        y = lam * y_onehot + (1 - lam) * y_onehot_perm
        return x, y, True
    elif cutmix_alpha > 0:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        indices = torch.randperm(x.size(0), device=x.device)
        
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        
        x[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        y_onehot_perm = y_onehot[indices]
        y = lam * y_onehot + (1 - lam) * y_onehot_perm
        return x, y, True
    
    return x, y, False


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, args, epoch, rank):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        # Mixup / CutMix
        x, y_mixed, is_mixed = mixup_cutmix(x, y, args.mixup_alpha, args.cutmix_alpha)
        
        optimizer.zero_grad()
        
        with autocast(enabled=args.amp):
            logits = model(x)
            if is_mixed:
                loss = -torch.sum(y_mixed * torch.log_softmax(logits, dim=1), dim=1).mean()
            else:
                loss = criterion(logits, y)
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        if not is_mixed:
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        
        # Log progress
        if rank == 0 and (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0
    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        with autocast(enabled=args.amp):
            logits = model(x)
            loss = criterion(logits, y)
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, path):
    """Save checkpoint."""
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["best_acc"]


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    
    # Setup distributed
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    device = torch.device("cuda")
    
    # Seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print("=" * 70)
        print("TwistNet ImageNet Pretraining")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print("=" * 70)
    
    # Data
    train_loader, val_loader, train_sampler = build_dataloaders(args, rank, world_size)
    
    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    twist_stages = tuple(int(x) for x in args.twist_stages.split(","))
    model = TwistNet(
        layers=[2, 2, 2, 2],
        num_classes=1000,
        twist_stages=twist_stages,
        num_heads=args.num_heads,
        gate_init=args.gate_init,
    ).to(device)
    
    if rank == 0:
        print(f"Model: TwistNet-18, Params: {count_params(model)/1e6:.2f}M")
    
    # Optional: torch.compile
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        if rank == 0:
            print("Using torch.compile")
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True
    )
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler() if args.amp else None
    
    # Resume
    start_epoch = 1
    best_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, scaler, args.resume, device)
        start_epoch += 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch-1}, best_acc: {best_acc:.4f}")
    
    # Training loop
    log_file = checkpoint_dir / "log.jsonl"
    
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, args, epoch, rank)
        val_loss, val_acc = validate(model, val_loader, criterion, args)
        scheduler.step()
        elapsed = time.time() - t0
        
        # Reduce metrics across GPUs
        if world_size > 1:
            metrics = torch.tensor([val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_acc = metrics[0].item() / world_size
        
        lr = optimizer.param_groups[0]["lr"]
        
        if rank == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | LR {lr:.6f} | "
                  f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                  f"Val Acc {val_acc*100:.2f}% | {elapsed:.1f}s")
            
            # Log
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "lr": lr,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                }) + "\n")
            
            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc,
                               checkpoint_dir / "best.pt")
                print(f"  New best: {best_acc*100:.2f}%")
            
            # Save periodic
            if epoch % args.save_freq == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc,
                               checkpoint_dir / f"epoch_{epoch}.pt")
            
            # Save latest (for resume)
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc,
                           checkpoint_dir / "latest.pt")
    
    # Final save
    if rank == 0:
        # Save final model weights to standard location
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        final_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        
        # Save to both checkpoint_dir and weights/
        torch.save(final_state, checkpoint_dir / "twistnet18_imagenet.pt")
        torch.save(final_state, weights_dir / "twistnet18_imagenet.pt")
        
        print("=" * 70)
        print(f"Training completed!")
        print(f"Best accuracy: {best_acc*100:.2f}%")
        print(f"Weights saved to:")
        print(f"  - {checkpoint_dir / 'twistnet18_imagenet.pt'}")
        print(f"  - {weights_dir / 'twistnet18_imagenet.pt'} (auto-detected by build_model)")
        print("=" * 70)
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
