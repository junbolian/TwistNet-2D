#!/usr/bin/env python3
"""Training script for TwistNet-2D benchmarks."""

import argparse
import json
import random
import time
import sys
import os
from pathlib import Path

# Add script directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import build_model, count_params
from datasets import get_dataloaders
from transforms import build_train_transform, build_eval_transform


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy_top1(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def mixup_cutmix(x, y, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0):
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


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dtd")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    # Model
    parser.add_argument("--model", type=str, default="twistnet18")
    parser.add_argument("--twist_stages", type=str, default="3,4")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--use_ais", action="store_true", default=True)
    parser.add_argument("--no_ais", action="store_true")
    parser.add_argument("--use_spiral", action="store_true", default=True)
    parser.add_argument("--no_spiral", action="store_true")
    parser.add_argument("--gate_init", type=float, default=-2.0)
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Augmentation
    parser.add_argument("--use_mixup", action="store_true", default=True)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ra_n", type=int, default=2)
    parser.add_argument("--ra_m", type=int, default=9)
    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--run_dir", type=str, default="runs")
    args = parser.parse_args()
    
    # Handle negative flags
    if args.no_ais: args.use_ais = False
    if args.no_spiral: args.use_spiral = False
    if args.no_mixup: args.use_mixup = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    twist_stages = tuple(int(x) for x in args.twist_stages.split(",")) if args.twist_stages else ()
    
    run_name = f"{args.dataset}_fold{args.fold}_{args.model}_seed{args.seed}"
    run_path = Path(args.run_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"{'='*60}")
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Data
    train_transform = build_train_transform(args.img_size, ra_n=args.ra_n, ra_m=args.ra_m)
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
    
    # Training
    best_val_acc = 0.0
    log_file = run_path / "log.jsonl"
    
    for epoch in range(1, args.epochs + 1):
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_path / "best.pt")
    
    # Test
    model.load_state_dict(torch.load(run_path / "best.pt"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Best Val: {best_val_acc:.4f}, Test: {test_acc:.4f}")
    print(f"{'='*60}")
    
    results = {"model": args.model, "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
               "best_val_acc": best_val_acc, "test_acc": test_acc, "params_M": count_params(model)/1e6}
    with open(run_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()