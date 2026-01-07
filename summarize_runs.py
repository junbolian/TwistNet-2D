#!/usr/bin/env python3
"""
Summarize experiment results from run directories.

Usage:
    python summarize_runs.py --run_dir runs/main --latex
    python summarize_runs.py --run_dir runs/ablation --csv
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_results(run_dir: Path):
    """Load all results.json files from run directory."""
    results = []
    for result_file in run_dir.glob("*/results.json"):
        with open(result_file) as f:
            data = json.load(f)
            data["run_name"] = result_file.parent.name
            results.append(data)
    return results


def aggregate_results(results):
    """Aggregate results by (dataset, model) across folds and seeds."""
    grouped = defaultdict(list)
    
    for r in results:
        key = (r["dataset"], r["model"])
        grouped[key].append({
            "fold": r.get("fold", 1),
            "seed": r.get("seed", 42),
            "test_acc": r["test_acc"],
            "best_val_acc": r.get("best_val_acc", r["test_acc"]),
            "params_M": r.get("params_M", 0),
        })
    
    aggregated = {}
    for (dataset, model), runs in grouped.items():
        test_accs = [r["test_acc"] for r in runs]
        val_accs = [r["best_val_acc"] for r in runs]
        params = runs[0]["params_M"] if runs else 0
        
        aggregated[(dataset, model)] = {
            "test_mean": np.mean(test_accs) * 100,
            "test_std": np.std(test_accs) * 100,
            "val_mean": np.mean(val_accs) * 100,
            "val_std": np.std(val_accs) * 100,
            "n_runs": len(runs),
            "params_M": params,
        }
    
    return aggregated


def print_table(aggregated, format="text"):
    """Print results table."""
    if not aggregated:
        print("No results found.")
        return
    
    # Group by dataset
    datasets = sorted(set(k[0] for k in aggregated.keys()))
    models = sorted(set(k[1] for k in aggregated.keys()))
    
    if format == "latex":
        print_latex_table(aggregated, datasets, models)
    elif format == "csv":
        print_csv_table(aggregated, datasets, models)
    else:
        print_text_table(aggregated, datasets, models)


def print_text_table(aggregated, datasets, models):
    """Print plain text table."""
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n### {dataset.upper()} ###")
        print(f"{'Model':<25} {'Params':<10} {'Test Acc':<20} {'N'}")
        print("-" * 60)
        
        # Sort by test accuracy
        dataset_results = [(m, aggregated.get((dataset, m))) for m in models 
                          if (dataset, m) in aggregated]
        dataset_results.sort(key=lambda x: x[1]["test_mean"], reverse=True)
        
        for model, stats in dataset_results:
            acc_str = f"{stats['test_mean']:.2f} ± {stats['test_std']:.2f}%"
            print(f"{model:<25} {stats['params_M']:.1f}M     {acc_str:<20} {stats['n_runs']}")
    
    print("\n" + "=" * 80)


def print_latex_table(aggregated, datasets, models):
    """Print LaTeX table."""
    print("\n% LaTeX Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Results on texture and fine-grained benchmarks.}")
    print("\\label{tab:results}")
    print("\\small")
    
    # Header
    cols = "l" + "c" * len(datasets)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    header = "Model & " + " & ".join([d.upper() for d in datasets]) + " \\\\"
    print(header)
    print("\\midrule")
    
    # Find best for each dataset
    best = {}
    for dataset in datasets:
        dataset_results = [(m, aggregated.get((dataset, m))) for m in models 
                          if (dataset, m) in aggregated]
        if dataset_results:
            best[dataset] = max(dataset_results, key=lambda x: x[1]["test_mean"])[0]
    
    # Rows
    for model in models:
        row = model.replace("_", "\\_")
        for dataset in datasets:
            if (dataset, model) in aggregated:
                stats = aggregated[(dataset, model)]
                acc = f"{stats['test_mean']:.1f}"
                if model == best.get(dataset):
                    acc = f"\\textbf{{{acc}}}"
                row += f" & {acc}"
            else:
                row += " & -"
        row += " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_csv_table(aggregated, datasets, models):
    """Print CSV table."""
    print("model," + ",".join(datasets))
    for model in models:
        row = model
        for dataset in datasets:
            if (dataset, model) in aggregated:
                stats = aggregated[(dataset, model)]
                row += f",{stats['test_mean']:.2f}"
            else:
                row += ","
        print(row)


def print_summary_stats(aggregated):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    total_runs = sum(s["n_runs"] for s in aggregated.values())
    datasets = set(k[0] for k in aggregated.keys())
    models = set(k[1] for k in aggregated.keys())
    
    print(f"Total runs: {total_runs}")
    print(f"Datasets: {len(datasets)}")
    print(f"Models: {len(models)}")
    
    # Best model per dataset
    print("\nBest model per dataset:")
    for dataset in sorted(datasets):
        dataset_results = [(m, aggregated.get((dataset, m))) for m in models 
                          if (dataset, m) in aggregated]
        if dataset_results:
            best_model, best_stats = max(dataset_results, key=lambda x: x[1]["test_mean"])
            print(f"  {dataset}: {best_model} ({best_stats['test_mean']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX format")
    parser.add_argument("--csv", action="store_true", help="Output CSV format")
    parser.add_argument("--summary", action="store_true", help="Print summary stats only")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
    
    results = load_results(run_dir)
    if not results:
        print(f"No results found in {run_dir}")
        return
    
    print(f"Loaded {len(results)} results from {run_dir}")
    
    aggregated = aggregate_results(results)
    
    if args.summary:
        print_summary_stats(aggregated)
    elif args.latex:
        print_table(aggregated, format="latex")
    elif args.csv:
        print_table(aggregated, format="csv")
    else:
        print_table(aggregated, format="text")
        print_summary_stats(aggregated)


if __name__ == "__main__":
    main()
