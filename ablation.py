#!/usr/bin/env python3
"""
Ablation study experiments for TwistNet-2D.

All models trained from scratch (no ImageNet pretraining).

Ablations:
1. Components: with/without AIS, with/without Spiral
2. Number of heads: 1, 2, 4
3. Twist stages: (3,), (4,), (3,4)
4. Gate initialization: -3, -2, -1, 0, 1
"""

import subprocess
import sys
import argparse
from pathlib import Path
from itertools import product


def is_completed(run_dir: str, config: dict, dataset: str, fold: int, seed: int) -> bool:
    """Check if ablation experiment is completed."""
    name = "_".join(f"{k}={v}" for k, v in sorted(config.items()))
    run_name = f"{dataset}_fold{fold}_{name}_seed{seed}"
    results_file = Path(run_dir) / run_name / "results.json"
    return results_file.exists()


def run_exp(config: dict, data_dir: str, dataset: str = "dtd", fold: int = 1,
            seed: int = 42, epochs: int = 200, run_dir: str = "runs_ablation", 
            dry_run: bool = False, force: bool = False):
    
    name = "_".join(f"{k}={v}" for k, v in sorted(config.items()))
    
    # Check if already completed
    if not force and is_completed(run_dir, config, dataset, fold, seed):
        print(f"[SKIP] {name} (already completed)")
        return
    
    cmd = [
        sys.executable, "train.py",
        "--data_dir", data_dir, "--dataset", dataset,
        "--fold", str(fold), "--seed", str(seed),
        "--model", "twistnet18", "--epochs", str(epochs),
        "--run_dir", run_dir, "--amp",
    ]
    
    if "num_heads" in config:
        cmd.extend(["--num_heads", str(config["num_heads"])])
    if "twist_stages" in config:
        cmd.extend(["--twist_stages", config["twist_stages"]])
    if "use_ais" in config and not config["use_ais"]:
        cmd.append("--no_ais")
    if "use_spiral" in config and not config["use_spiral"]:
        cmd.append("--no_spiral")
    if "gate_init" in config:
        cmd.extend(["--gate_init", str(config["gate_init"])])
    
    # Check for checkpoint
    run_name = f"{dataset}_fold{fold}_{name}_seed{seed}"
    checkpoint = Path(run_dir) / run_name / "checkpoint.pt"
    if checkpoint.exists():
        cmd.append("--resume")
        print(f"\n[RESUME] {name}")
    else:
        print(f"\n[NEW] {name}")
    
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] {e}")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            sys.exit(1)


def ablation_num_heads(data_dir: str, **kwargs):
    """Ablation: Number of spiral heads."""
    print("\n" + "=" * 60)
    print("Ablation: Number of Heads (1, 2, 4)")
    print("=" * 60)
    for num_heads in [1, 2, 4]:
        run_exp({"num_heads": num_heads}, data_dir, **kwargs)


def ablation_twist_stages(data_dir: str, **kwargs):
    """Ablation: Which stages to add TwistBlock."""
    print("\n" + "=" * 60)
    print("Ablation: Twist Stages")
    print("=" * 60)
    for stages in ["3", "4", "3,4"]:
        run_exp({"twist_stages": stages}, data_dir, **kwargs)


def ablation_components(data_dir: str, **kwargs):
    """Ablation: AIS and Spiral components."""
    print("\n" + "=" * 60)
    print("Ablation: Components (AIS, Spiral)")
    print("=" * 60)
    configs = [
        {"use_ais": True, "use_spiral": True},   # Full
        {"use_ais": True, "use_spiral": False},  # No spiral
        {"use_ais": False, "use_spiral": True},  # No AIS
        {"use_ais": False, "use_spiral": False}, # Baseline (local pairwise only)
    ]
    for cfg in configs:
        run_exp(cfg, data_dir, **kwargs)


def ablation_gate_init(data_dir: str, **kwargs):
    """Ablation: Gate initialization values."""
    print("\n" + "=" * 60)
    print("Ablation: Gate Initialization")
    print("=" * 60)
    for gate_init in [-3.0, -2.0, -1.0, 0.0, 1.0]:
        run_exp({"gate_init": gate_init}, data_dir, **kwargs)


def run_all_ablations(data_dir: str, dataset: str = "dtd", fold: int = 1,
                      seed: int = 42, epochs: int = 200, dry_run: bool = False,
                      force: bool = False):
    """Run all ablation studies."""
    kwargs = {"dataset": dataset, "fold": fold, "seed": seed, "epochs": epochs, 
              "dry_run": dry_run, "force": force}
    
    print("=" * 60)
    print("TwistNet-2D Ablation Studies")
    print(f"Dataset: {dataset}, Fold: {fold}, Seed: {seed}")
    print("Training: From scratch (no pretraining)")
    print("=" * 60)
    
    ablation_components(data_dir, **kwargs)
    ablation_num_heads(data_dir, **kwargs)
    ablation_twist_stages(data_dir, **kwargs)
    ablation_gate_init(data_dir, **kwargs)
    
    print("\n" + "=" * 60)
    print("All ablation studies completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dtd")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force re-run completed experiments")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "heads", "stages", "components", "gate"])
    args = parser.parse_args()
    
    funcs = {
        "all": run_all_ablations,
        "heads": ablation_num_heads,
        "stages": ablation_twist_stages,
        "components": ablation_components,
        "gate": ablation_gate_init,
    }
    
    kwargs = {"dataset": args.dataset, "fold": args.fold, "seed": args.seed,
              "epochs": args.epochs, "dry_run": args.dry_run, "force": args.force}
    funcs[args.ablation](args.data_dir, **kwargs)
