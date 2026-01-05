#!/usr/bin/env python3
"""
run_all.py - Batch experiment runner for TwistNet-2D benchmarks.

Usage:
    python run_all.py --data_dir data/dtd --dataset dtd --folds 1-10 --seeds 42,43,44 \
        --models resnet18,twistnet18 --epochs 200
"""

import argparse
import subprocess
import sys
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


def main():
    ap = argparse.ArgumentParser(description="Batch experiment runner")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    ap.add_argument("--dataset", type=str, default="dtd", help="Dataset name")
    ap.add_argument("--folds", type=str, default="1", help="Fold range (e.g., '1-10' or '1,2,3')")
    ap.add_argument("--seeds", type=str, default="42", help="Seed list (e.g., '42,43,44')")
    ap.add_argument("--models", type=str, default="resnet18,twistnet18", help="Model list")
    ap.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    ap.add_argument("--img_size", type=int, default=224, help="Image size")
    ap.add_argument("--run_dir", type=str, default="runs", help="Output directory")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = ap.parse_args()
    
    folds = parse_range(args.folds)
    seeds = parse_range(args.seeds)
    models = [m.strip() for m in args.models.split(",")]
    
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
    print(f"Total runs: {total}")
    print("=" * 60)
    
    count, failed = 0, []
    
    for model, fold, seed in product(models, folds, seeds):
        count += 1
        run_name = f"{args.dataset}_fold{fold}_{model}_seed{seed}"
        print(f"\n[{count}/{total}] {run_name}")
        
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
        ]
        
        if args.dry_run:
            print(f"  [DRY] {' '.join(cmd)}")
        else:
            try:
                # Use cwd to ensure modules are found
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                subprocess.run(cmd, check=True, cwd=script_dir)
            except subprocess.CalledProcessError as e:
                print(f"  [FAILED] {e}")
                failed.append(run_name)
            except KeyboardInterrupt:
                print("\n[INTERRUPTED by user]")
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print(f"Completed: {count - len(failed)}/{count}")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()