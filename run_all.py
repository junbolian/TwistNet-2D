#!/usr/bin/env python3
"""
run_all.py - Batch experiment runner for TwistNet-2D benchmarks.

Features:
- Automatically skips completed experiments (checks results.json)
- Resume from checkpoint for interrupted experiments
- Parallel-safe: multiple instances can run different experiments

Usage:
    python run_all.py --data_dir data/dtd --dataset dtd --folds 1-10 --seeds 42,43,44 \
        --models resnet18,twistnet18 --epochs 100
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from itertools import product


def parse_range(s: str):
    """Parse range string like '1-10' or '1,2,3' into list of ints."""
    result = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


VALID_MODELS = [
    # Group 1: Fair comparison (10-16M params) - MAIN
    "resnet18", "seresnet18", "convnextv2_nano", "convnextv2_pico",
    "fastvit_sa12", "efficientformerv2_s1", "repvit_m1_5",
    "twistnet18",
    # Ablation
    "twistnet18_no_spiral", "twistnet18_no_ais", "twistnet18_first_order",
    # Group 2: Efficiency comparison (official tiny ~25-30M)
    "convnext_tiny", "convnextv2_tiny", "swin_tiny", "maxvit_tiny",
    # Group 3: Additional
    "efficientnet_b0", "efficientnetv2_s", "mobilenetv3_large", "regnety_016",
]


def is_experiment_completed(run_dir: Path, run_name: str) -> bool:
    """Check if experiment is already completed (has results.json)."""
    results_file = run_dir / run_name / "results.json"
    return results_file.exists()


def has_checkpoint(run_dir: Path, run_name: str) -> bool:
    """Check if experiment has a checkpoint to resume from."""
    checkpoint_file = run_dir / run_name / "checkpoint.pt"
    return checkpoint_file.exists()


def main():
    ap = argparse.ArgumentParser(description="Batch experiment runner")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    ap.add_argument("--dataset", type=str, default="dtd", help="Dataset name")
    ap.add_argument("--folds", type=str, default="1", help="Fold range (e.g., '1-10' or '1,2,3')")
    ap.add_argument("--seeds", type=str, default="42", help="Seed list (e.g., '42,43,44')")
    ap.add_argument("--models", type=str, default="resnet18,twistnet18", help="Model list")
    ap.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    ap.add_argument("--img_size", type=int, default=224, help="Image size")
    ap.add_argument("--run_dir", type=str, default="runs", help="Output directory")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    ap.add_argument("--force", action="store_true", help="Force re-run even if completed")
    args = ap.parse_args()
    
    folds = parse_range(args.folds)
    seeds = parse_range(args.seeds)
    models = [m.strip() for m in args.models.split(",")]
    run_dir = Path(args.run_dir)
    
    # Validate models
    for m in models:
        if m not in VALID_MODELS:
            print(f"[ERROR] Unknown model: {m}")
            print(f"Valid models: {VALID_MODELS}")
            sys.exit(1)
    
    total = len(models) * len(folds) * len(seeds)
    
    print("=" * 60)
    print("TwistNet-2D Batch Experiment Runner")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {models}")
    print(f"Folds: {folds}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Total experiments: {total}")
    print("=" * 60)
    
    # Count status
    completed = 0
    to_run = []
    
    for model, fold, seed in product(models, folds, seeds):
        run_name = f"{args.dataset}_fold{fold}_{model}_seed{seed}"
        if is_experiment_completed(run_dir, run_name) and not args.force:
            completed += 1
        else:
            to_run.append((model, fold, seed, run_name))
    
    print(f"Already completed: {completed}/{total}")
    print(f"To run: {len(to_run)}/{total}")
    print("=" * 60)
    
    if not to_run:
        print("All experiments completed!")
        return
    
    count, failed = 0, []
    
    for model, fold, seed, run_name in to_run:
        count += 1
        
        # Check if can resume
        resume = has_checkpoint(run_dir, run_name)
        status = "[RESUME]" if resume else "[NEW]"
        print(f"\n{status} [{count}/{len(to_run)}] {run_name}")
        
        cmd = [
            sys.executable, "train.py",
            "--data_dir", args.data_dir,
            "--dataset", args.dataset,
            "--fold", str(fold),
            "--seed", str(seed),
            "--model", model,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--img_size", str(args.img_size),
            "--run_dir", args.run_dir,
            "--amp",
            "--pretrained",  # CRITICAL: use pretrained weights
        ]
        
        if resume:
            cmd.append("--resume")
        
        if args.dry_run:
            print(f"  [DRY] {' '.join(cmd)}")
        else:
            try:
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                subprocess.run(cmd, check=True, cwd=script_dir)
            except subprocess.CalledProcessError as e:
                print(f"  [FAILED] {e}")
                failed.append(run_name)
            except KeyboardInterrupt:
                print("\n[INTERRUPTED by user]")
                print("Run again to resume from checkpoint.")
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print(f"Completed: {count - len(failed)}/{len(to_run)}")
    print(f"Previously completed: {completed}")
    print(f"Total: {completed + count - len(failed)}/{total}")
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
