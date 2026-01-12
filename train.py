#!/usr/bin/env python3
"""
Training script for TwistNet-2D benchmarks.
All models trained from scratch (no ImageNet pretraining).
Supports: checkpoint resume, mixed precision, dataset-specific augmentation.

Key parameters (matching the original high-accuracy version):
- lr: 0.05 (NOT 0.01)
- batch_size: 64 (NOT 32)
- min_lr: 1e-5 (NOT 1e-6)
- crop_scale: (0.25, 1.0) for texture datasets (improvement over 0.08)
"""

import argparse
import json
import random
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import build_model, count_params
from datasets import get_dataloaders
from transforms import build_train_transform, build_eval_transform, get_dataset_transform_config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy_top1(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def mixup_cutmix(x, y, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0):
    """Mixup and CutMix augmentation."""
    if random.random() > prob:
        return x, y, y, 1.0
    
    bs = x.size(0)
    indices = torch.randperm(bs, device=x.device)
    
    if random.random() < 0.5 and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)
        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        x_mixed = lam * x + (1 - lam) * x[indices]
    
    return x_mixed, y, y[indices], lam


def train_one_epoch(model, loader, optimizer, device, scaler, args):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_loss, total_acc, n = 0., 0., 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        if args.use_mixup:
            x, y_a, y_b, lam = mixup_cutmix(x, y, args.mixup_alpha, args.cutmix_alpha)
        
        with autocast(enabled=args.amp):
            logits = model(x)
            if args.use_mixup and lam < 1.0:
                loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
            else:
                loss = loss_fn(logits, y)
        
        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits.detach(), y) * bs
        n += bs
    
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0., 0., 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs
    
    return total_loss / n, total_acc / n


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_acc):
    """Save checkpoint for resuming training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint for resuming training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_acc']


def main():
    parser = argparse.ArgumentParser(description="TwistNet-2D Training (From Scratch)")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--dataset", type=str, default="dtd", 
                        choices=["dtd", "fmd", "kth_tips2", "cub200", "flowers102"])
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    
    # Model
    parser.add_argument("--model", type=str, default="twistnet18", help="Model name")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use ImageNet pretrained weights (default: False)")
    parser.add_argument("--twist_stages", type=str, default="3,4", help="Stages to use TwistBlock")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of STCI heads")
    parser.add_argument("--use_ais", action="store_true", default=True)
    parser.add_argument("--no_ais", action="store_true")
    parser.add_argument("--use_spiral", action="store_true", default=True)
    parser.add_argument("--no_spiral", action="store_true")
    parser.add_argument("--gate_init", type=float, default=-2.0, help="Gate initialization")
    
    # Training - KEY PARAMETERS (matching original high-accuracy version)
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (was 32, now 64)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (was 0.01, now 0.05)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR (was 1e-6, now 1e-5)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Augmentation
    parser.add_argument("--use_mixup", action="store_true", default=True)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.8, help="Mixup alpha")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ra_n", type=int, default=2, help="RandAugment num ops")
    parser.add_argument("--ra_m", type=int, default=9, help="RandAugment magnitude")
    parser.add_argument("--crop_scale_min", type=float, default=0.2, help="Min crop scale (unified 0.2)")
    parser.add_argument("--crop_scale_max", type=float, default=1.0, help="Max crop scale")
    parser.add_argument("--auto_augment", action="store_true", default=True,
                        help="Use dataset-specific augmentation settings")
    parser.add_argument("--no_auto_augment", action="store_true")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--run_dir", type=str, default="runs", help="Output directory")
    
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    args = parser.parse_args()
    
    # Handle negative flags
    if args.no_ais: args.use_ais = False
    if args.no_spiral: args.use_spiral = False
    if args.no_mixup: args.use_mixup = False
    if args.no_auto_augment: args.auto_augment = False
    
    # Apply dataset-specific augmentation settings
    if args.auto_augment:
        config = get_dataset_transform_config(args.dataset)
        args.crop_scale_min = config["crop_scale"][0]
        args.crop_scale_max = config["crop_scale"][1]
        args.ra_n = config["ra_n"]
        args.ra_m = config["ra_m"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    twist_stages = tuple(int(x) for x in args.twist_stages.split(",")) if args.twist_stages else ()
    
    # Run name and path
    run_name = f"{args.dataset}_fold{args.fold}_{args.model}_seed{args.seed}"
    run_path = Path(args.run_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    results_file = run_path / "results.json"
    if results_file.exists() and not args.resume:
        print(f"[SKIP] {run_name} already completed. Use --resume to continue.")
        with open(results_file) as f:
            results = json.load(f)
        print(f"  Best Val: {results.get('best_val_acc', 'N/A'):.4f}, Test: {results.get('test_acc', 'N/A'):.4f}")
        return
    
    # Save config
    with open(run_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"{'='*60}")
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Training: From scratch (no pretraining)")
    print(f"{'='*60}")
    print(f"Key settings:")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"  Crop scale: ({args.crop_scale_min}, {args.crop_scale_max})")
    print(f"  Mixup: alpha={args.mixup_alpha}, CutMix: alpha={args.cutmix_alpha}")
    print(f"  RandAugment: n={args.ra_n}, m={args.ra_m}")
    print(f"{'='*60}")
    
    # Data - use updated transform
    crop_scale = (args.crop_scale_min, args.crop_scale_max)
    train_transform = build_train_transform(
        args.img_size, 
        use_randaugment=True,
        ra_n=args.ra_n, 
        ra_m=args.ra_m,
        crop_scale=crop_scale
    )
    eval_transform = build_eval_transform(args.img_size)
    
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.data_dir, args.dataset, args.fold, train_transform, eval_transform,
        args.batch_size, args.num_workers, args.seed
    )
    
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    model = build_model(
        args.model, num_classes,
        pretrained=args.pretrained,
        twist_stages=twist_stages, num_heads=args.num_heads,
        use_ais=args.use_ais, use_spiral=args.use_spiral, gate_init=args.gate_init
    ).to(device)
    
    print(f"Model: {args.model}, Params: {count_params(model)/1e6:.2f}M")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 weight_decay=args.weight_decay, nesterov=True)
    
    if args.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    scaler = GradScaler() if args.amp else None
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    checkpoint_path = args.checkpoint or (run_path / "checkpoint.pt")
    
    if args.resume and Path(checkpoint_path).exists():
        print(f"[RESUME] Loading checkpoint from {checkpoint_path}")
        start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"[RESUME] Continuing from epoch {start_epoch}, best_val_acc: {best_val_acc:.4f}")
    
    # Training
    log_file = run_path / "log.jsonl"
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, args)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        
        log = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        
        if hasattr(model, "get_gate_values"):
            gates = model.get_gate_values()
            if gates:
                log["gates"] = {k: round(v, 4) for k, v in list(gates.items())[:4]}
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log) + "\n")
        
        print(f"Epoch {epoch:3d}/{args.epochs} | LR {lr:.5f} | Train {train_acc:.4f} | Val {val_acc:.4f} | {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_path / "best.pt")
        
        # Save checkpoint for resume
        save_checkpoint(run_path / "checkpoint.pt", model, optimizer, scheduler, scaler, epoch, best_val_acc)
    
    # Test
    model.load_state_dict(torch.load(run_path / "best.pt"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Best Val: {best_val_acc:.4f}, Test: {test_acc:.4f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "model": args.model, 
        "dataset": args.dataset, 
        "fold": args.fold, 
        "seed": args.seed,
        "best_val_acc": best_val_acc, 
        "test_acc": test_acc, 
        "params_M": count_params(model)/1e6,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "crop_scale": (args.crop_scale_min, args.crop_scale_max),
    }
    with open(run_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Clean up checkpoint
    checkpoint_file = run_path / "checkpoint.pt"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


if __name__ == "__main__":
    main()