#!/usr/bin/env python3
"""
summarize_runs.py - Summarize experiment results and generate paper tables.

Usage:
    python summarize_runs.py --run_dir runs --dataset dtd
    python summarize_runs.py --run_dir runs --all_datasets
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(run_dir: Path, dataset: str = None):
    """Load all results from a run directory."""
    results = defaultdict(lambda: {"test_acc": [], "val_acc": [], "params": 0})
    
    for run_path in run_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        # Filter by dataset if specified
        if dataset and not run_path.name.startswith(dataset):
            continue
        
        results_file = run_path / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            model = data["model"]
            results[model]["test_acc"].append(data["test_acc"])
            results[model]["val_acc"].append(data.get("best_val_acc", 0))
            results[model]["params"] = data.get("params_M", 0)
            results[model]["dataset"] = data.get("dataset", dataset)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {results_file}: {e}")
    
    return results


def print_summary_table(results: dict, dataset: str):
    """Print formatted summary table."""
    if not results:
        print(f"No results found for {dataset}")
        return
    
    # Sort by model order
    order = ["resnet18", "se_resnet18", "convnext", "hornet", "focalnet", "van", "moganet", "twistnet18"]
    models = sorted(results.keys(), key=lambda x: order.index(x) if x in order else 99)
    
    print(f"\n{'='*75}")
    print(f"Results Summary: {dataset.upper()}")
    print(f"{'='*75}")
    print(f"{'Model':<15} {'Params (M)':<12} {'Test Acc (%)':<20} {'Val Acc (%)':<15} {'Runs'}")
    print("-" * 75)
    
    best_acc = 0
    best_model = ""
    
    for model in models:
        data = results[model]
        test_accs = np.array(data["test_acc"]) * 100
        val_accs = np.array(data["val_acc"]) * 100 if data["val_acc"] else test_accs
        params = data["params"]
        n = len(test_accs)
        
        if n == 0:
            continue
        
        mean_test = test_accs.mean()
        std_test = test_accs.std()
        mean_val = val_accs.mean()
        
        if mean_test > best_acc:
            best_acc = mean_test
            best_model = model
        
        # Highlight TwistNet
        marker = "★" if model == "twistnet18" else " "
        print(f"{marker}{model:<14} {params:<12.2f} {mean_test:>6.2f} ± {std_test:<6.2f}    {mean_val:>6.2f}         {n}")
    
    print("-" * 75)
    print(f"Best model: {best_model} ({best_acc:.2f}%)")
    print(f"{'='*75}")


def generate_latex_table(results: dict, dataset: str) -> str:
    """Generate LaTeX table for paper."""
    order = ["resnet18", "se_resnet18", "convnext", "hornet", "focalnet", "van", "moganet", "twistnet18"]
    models = sorted(results.keys(), key=lambda x: order.index(x) if x in order else 99)
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Results on " + dataset.upper() + r" dataset. Best results in \textbf{bold}.}",
        r"\label{tab:" + dataset + r"}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Params (M) & Accuracy (\%) \\",
        r"\midrule",
    ]
    
    best_acc = max(np.array(results[m]["test_acc"]).mean() for m in models if results[m]["test_acc"])
    
    for model in models:
        data = results[model]
        if not data["test_acc"]:
            continue
        
        test_accs = np.array(data["test_acc"]) * 100
        params = data["params"]
        mean = test_accs.mean()
        std = test_accs.std()
        
        # Format model name
        name_map = {
            "resnet18": "ResNet-18",
            "se_resnet18": "SE-ResNet-18",
            "convnext": "ConvNeXt-F",
            "hornet": "HorNet-T",
            "focalnet": "FocalNet-T",
            "van": "VAN-B1",
            "moganet": "MogaNet-XT",
            "twistnet18": "TwistNet-18",
        }
        display_name = name_map.get(model, model)
        
        # Bold for best or TwistNet
        if abs(mean - best_acc * 100) < 0.01 or model == "twistnet18":
            lines.append(rf"\textbf{{{display_name}}} & {params:.1f} & \textbf{{{mean:.2f} $\pm$ {std:.2f}}} \\")
        else:
            lines.append(rf"{display_name} & {params:.1f} & {mean:.2f} $\pm$ {std:.2f} \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_comparison_table(all_results: dict) -> str:
    """Generate multi-dataset comparison table."""
    datasets = list(all_results.keys())
    models = ["resnet18", "se_resnet18", "convnext", "hornet", "focalnet", "van", "moganet", "twistnet18"]
    
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison across texture and fine-grained recognition datasets.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{l" + "c" * len(datasets) + r"}",
        r"\toprule",
        r"Model & " + " & ".join(d.upper() for d in datasets) + r" \\",
        r"\midrule",
    ]
    
    for model in models:
        name_map = {
            "resnet18": "ResNet-18",
            "se_resnet18": "SE-ResNet-18",
            "convnext": "ConvNeXt-F",
            "hornet": "HorNet-T",
            "focalnet": "FocalNet-T",
            "van": "VAN-B1",
            "moganet": "MogaNet-XT",
            "twistnet18": "\\textbf{TwistNet-18}",
        }
        display_name = name_map.get(model, model)
        
        accs = []
        for ds in datasets:
            if model in all_results[ds] and all_results[ds][model]["test_acc"]:
                mean = np.array(all_results[ds][model]["test_acc"]).mean() * 100
                std = np.array(all_results[ds][model]["test_acc"]).std() * 100
                if model == "twistnet18":
                    accs.append(rf"\textbf{{{mean:.1f}}}")
                else:
                    accs.append(f"{mean:.1f}")
            else:
                accs.append("-")
        
        lines.append(f"{display_name} & " + " & ".join(accs) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--run_dir", type=str, default="runs", help="Run directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (if single)")
    parser.add_argument("--all_datasets", action="store_true", help="Process all datasets")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--output", type=str, default=None, help="Output file for LaTeX")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory '{run_dir}' does not exist")
        return
    
    datasets = ["dtd", "fmd", "kth_tips2", "cub200", "flowers102"] if args.all_datasets else [args.dataset or "dtd"]
    
    all_results = {}
    latex_tables = []
    
    for dataset in datasets:
        # Check if dataset-specific subdirectory exists
        ds_dir = run_dir / dataset if (run_dir / dataset).exists() else run_dir
        results = load_results(ds_dir, dataset)
        
        if results:
            all_results[dataset] = results
            print_summary_table(results, dataset)
            
            if args.latex:
                latex_tables.append(generate_latex_table(results, dataset))
    
    # Generate comparison table
    if args.all_datasets and len(all_results) > 1:
        if args.latex:
            latex_tables.append(generate_comparison_table(all_results))
    
    # Save LaTeX output
    if args.latex and latex_tables:
        latex_output = "\n\n".join(latex_tables)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(latex_output)
            print(f"\nLaTeX tables saved to: {args.output}")
        else:
            print("\n" + "=" * 75)
            print("LaTeX Tables")
            print("=" * 75)
            print(latex_output)


if __name__ == "__main__":
    main()