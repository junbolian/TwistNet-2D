#!/usr/bin/env python3
"""
plot_results.py - Publication-Quality Visualization for TwistNet-2D

Generates all figures needed for a top-tier venue (ECCV/CVPR/ICCV):

1. Main Results
   - Bar chart comparing models across datasets
   - Radar chart for multi-dataset comparison
   - Params vs Accuracy scatter plot

2. Ablation Study
   - Component ablation bar chart
   - Gate value evolution

3. Efficiency Analysis
   - FLOPs vs Accuracy
   - Inference speed comparison

4. Qualitative Analysis
   - Grad-CAM attention maps
   - t-SNE feature visualization
   - Interaction matrix heatmaps

Usage:
    python plot_results.py --run_dir runs/main --save_dir figures
    python plot_results.py --run_dir runs/main --plot all
    python plot_results.py --run_dir runs/main --plot bar,radar,scatter
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# Optional imports
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from models import build_model, count_params
    from datasets import get_dataloaders
    from transforms import build_eval_transform
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Style Configuration (Nature/Science style)
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    'resnet18': '#4C72B0',
    'seresnet18': '#55A868',
    'convnextv2_nano': '#C44E52',
    'fastvit_sa12': '#8172B3',
    'repvit_m1_5': '#64B5CD',
    'twistnet18': '#DD8452',  # Our method - highlighted
    'convnext_tiny': '#937860',
    'swin_tiny': '#8C8C8C',
}

# Model display names
MODEL_NAMES = {
    'resnet18': 'ResNet-18',
    'seresnet18': 'SE-ResNet-18',
    'convnextv2_nano': 'ConvNeXtV2-N',
    'fastvit_sa12': 'FastViT-SA12',
    'repvit_m1_5': 'RepViT-M1.5',
    'twistnet18': 'TwistNet-18 (Ours)',
    'convnext_tiny': 'ConvNeXt-T',
    'swin_tiny': 'Swin-T',
}

DATASET_NAMES = {
    'dtd': 'DTD',
    'fmd': 'FMD',
    'cub200': 'CUB-200',
    'flowers102': 'Flowers-102',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# Data Loading
# =============================================================================

def load_results(run_dir: Path):
    """Load all results from run directory."""
    results = []
    for result_file in run_dir.glob("*/results.json"):
        with open(result_file) as f:
            data = json.load(f)
            data["run_name"] = result_file.parent.name
            results.append(data)
    return results


def aggregate_results(results):
    """Aggregate results by (dataset, model)."""
    grouped = defaultdict(list)
    
    for r in results:
        key = (r["dataset"], r["model"])
        grouped[key].append({
            "test_acc": r["test_acc"],
            "params_M": r.get("params_M", 0),
        })
    
    aggregated = {}
    for (dataset, model), runs in grouped.items():
        test_accs = [r["test_acc"] for r in runs]
        params = runs[0]["params_M"] if runs else 0
        
        aggregated[(dataset, model)] = {
            "mean": np.mean(test_accs) * 100,
            "std": np.std(test_accs) * 100,
            "n": len(runs),
            "params_M": params,
        }
    
    return aggregated


# =============================================================================
# 1. Bar Chart - Model Comparison per Dataset
# =============================================================================

def plot_bar_chart(aggregated, datasets=None, models=None, save_path="figures/bar_chart.pdf"):
    """Bar chart comparing models across datasets."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = sorted(set(k[0] for k in aggregated.keys()))
    if models is None:
        models = sorted(set(k[1] for k in aggregated.keys()))
    
    # Filter to available data
    datasets = [d for d in datasets if any((d, m) in aggregated for m in models)]
    models = [m for m in models if any((d, m) in aggregated for d in datasets)]
    
    n_datasets = len(datasets)
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(2 + 1.5 * n_datasets, 4))
    
    x = np.arange(n_datasets)
    width = 0.8 / n_models
    
    for i, model in enumerate(models):
        means = []
        stds = []
        for dataset in datasets:
            if (dataset, model) in aggregated:
                means.append(aggregated[(dataset, model)]["mean"])
                stds.append(aggregated[(dataset, model)]["std"])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_models / 2 + 0.5) * width
        color = COLORS.get(model, '#888888')
        label = MODEL_NAMES.get(model, model)
        
        # Highlight our method
        edgecolor = 'black' if model == 'twistnet18' else 'none'
        linewidth = 1.5 if model == 'twistnet18' else 0
        
        bars = ax.bar(x + offset, means, width * 0.9, label=label, 
                      color=color, edgecolor=edgecolor, linewidth=linewidth,
                      yerr=stds, capsize=2, error_kw={'linewidth': 0.8})
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.legend(loc='upper right', ncol=2, frameon=True, fancybox=False)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 2. Radar Chart - Multi-Dataset Comparison
# =============================================================================

def plot_radar_chart(aggregated, datasets=None, models=None, save_path="figures/radar_chart.pdf"):
    """Radar chart for multi-dataset performance comparison."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = sorted(set(k[0] for k in aggregated.keys()))
    if models is None:
        models = sorted(set(k[1] for k in aggregated.keys()))
    
    # Filter
    datasets = [d for d in datasets if any((d, m) in aggregated for m in models)]
    models = [m for m in models if any((d, m) in aggregated for d in datasets)]
    
    n_datasets = len(datasets)
    angles = np.linspace(0, 2 * np.pi, n_datasets, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for model in models:
        values = []
        for dataset in datasets:
            if (dataset, model) in aggregated:
                values.append(aggregated[(dataset, model)]["mean"])
            else:
                values.append(0)
        values += values[:1]  # Close
        
        color = COLORS.get(model, '#888888')
        label = MODEL_NAMES.get(model, model)
        linewidth = 2.5 if model == 'twistnet18' else 1.5
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=label, color=color)
        ax.fill(angles, values, alpha=0.1 if model != 'twistnet18' else 0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.set_ylim(50, 100)
    ax.set_yticks([60, 70, 80, 90])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 3. Params vs Accuracy Scatter Plot
# =============================================================================

def plot_params_vs_accuracy(aggregated, dataset='dtd', save_path="figures/params_accuracy.pdf"):
    """Scatter plot: Parameters vs Accuracy."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = sorted(set(k[1] for k in aggregated.keys() if k[0] == dataset))
    
    for model in models:
        key = (dataset, model)
        if key not in aggregated:
            continue
        
        stats = aggregated[key]
        params = stats["params_M"]
        acc = stats["mean"]
        std = stats["std"]
        
        color = COLORS.get(model, '#888888')
        label = MODEL_NAMES.get(model, model)
        
        # Marker size and style
        marker = '*' if model == 'twistnet18' else 'o'
        size = 300 if model == 'twistnet18' else 150
        
        ax.scatter(params, acc, s=size, c=color, marker=marker, label=label, 
                   edgecolors='black' if model == 'twistnet18' else 'none',
                   linewidths=1.5, zorder=10 if model == 'twistnet18' else 5)
        
        # Error bar
        ax.errorbar(params, acc, yerr=std, fmt='none', color=color, capsize=3, alpha=0.7)
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel(f'Test Accuracy on {DATASET_NAMES.get(dataset, dataset)} (%)')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add Pareto frontier hint
    ax.annotate('Better', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'),
                xytext=(0.15, 0.85))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 4. Ablation Bar Chart
# =============================================================================

def plot_ablation_chart(aggregated, dataset='dtd', save_path="figures/ablation.pdf"):
    """Bar chart for ablation study."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    ablation_models = [
        'twistnet18',
        'twistnet18_no_spiral', 
        'twistnet18_no_ais',
        'twistnet18_first_order'
    ]
    
    labels = [
        'TwistNet-18\n(Full)',
        'w/o Spiral\nTwist',
        'w/o AIS',
        'First-Order\nOnly'
    ]
    
    accs = []
    stds = []
    for model in ablation_models:
        key = (dataset, model)
        if key in aggregated:
            accs.append(aggregated[key]["mean"])
            stds.append(aggregated[key]["std"])
        else:
            accs.append(0)
            stds.append(0)
    
    if all(a == 0 for a in accs):
        print(f"No ablation data found for {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#DD8452', '#4C72B0', '#55A868', '#C44E52']
    x = np.arange(len(labels))
    
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1)
    
    # Highlight full model
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)
    
    # Add delta annotations
    baseline = accs[0]
    for i in range(1, len(accs)):
        if accs[i] > 0:
            delta = accs[i] - baseline
            ax.annotate(f'{delta:+.1f}%', xy=(i, accs[i] + stds[i] + 1),
                        ha='center', fontsize=10, color='red' if delta < 0 else 'green')
    
    ax.set_ylabel(f'Test Accuracy on {DATASET_NAMES.get(dataset, dataset)} (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(accs) * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 5. FLOPs vs Accuracy (Efficiency)
# =============================================================================

def plot_efficiency(save_path="figures/efficiency.pdf"):
    """Plot FLOPs vs Accuracy for efficiency comparison."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping efficiency plot")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Compute FLOPs for each model
    models_to_test = [
        'resnet18', 'seresnet18', 'convnextv2_nano', 
        'fastvit_sa12', 'twistnet18',
        'convnext_tiny', 'swin_tiny'
    ]
    
    results = {}
    x = torch.randn(1, 3, 224, 224)
    
    for model_name in models_to_test:
        try:
            model = build_model(model_name, num_classes=47, pretrained=False)
            model.eval()
            
            params = count_params(model) / 1e6
            
            # Estimate FLOPs using forward pass time as proxy
            # (Real FLOPs would need torchprofile or fvcore)
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    _ = model(x)
                
                # Time
                start = time.time()
                for _ in range(20):
                    _ = model(x)
                elapsed = (time.time() - start) / 20 * 1000  # ms
            
            results[model_name] = {'params': params, 'time_ms': elapsed}
            print(f"  {model_name}: {params:.1f}M, {elapsed:.1f}ms")
            
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")
    
    if not results:
        print("No models tested successfully")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model, stats in results.items():
        color = COLORS.get(model, '#888888')
        label = MODEL_NAMES.get(model, model)
        marker = '*' if model == 'twistnet18' else 'o'
        size = 300 if model == 'twistnet18' else 150
        
        ax.scatter(stats['params'], stats['time_ms'], s=size, c=color, 
                   marker=marker, label=label,
                   edgecolors='black' if model == 'twistnet18' else 'none',
                   linewidths=1.5)
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Inference Time (ms)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 6. t-SNE Feature Visualization
# =============================================================================

@torch.no_grad()
def plot_tsne(model, dataloader, num_classes=10, num_samples=500, 
              save_path="figures/tsne.pdf"):
    """t-SNE visualization of learned features."""
    if not SKLEARN_AVAILABLE or not TORCH_AVAILABLE:
        print("sklearn or torch not available, skipping t-SNE")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    features = []
    labels = []
    
    for x, y in dataloader:
        x = x.to(device)
        
        # Get penultimate layer features
        feat = model.get_features(x)['layer4']
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        
        features.append(feat.cpu().numpy())
        labels.append(y.numpy())
        
        if sum(len(f) for f in features) >= num_samples:
            break
    
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Filter to num_classes
    mask = labels < num_classes
    features = features[mask]
    labels = labels[mask]
    
    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.cm.get_cmap('tab10', num_classes)
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels, 
                         cmap=cmap, alpha=0.7, s=20)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE of TwistNet-18 Features')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(num_classes))
    cbar.set_label('Class')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 7. Gate Value Evolution (from training log)
# =============================================================================

def plot_gate_evolution(log_file, save_path="figures/gate_evolution.pdf"):
    """Plot gate value evolution during training."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    epochs = []
    gates = {}
    
    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            if 'gates' in data:
                for name, val in data['gates'].items():
                    if name not in gates:
                        gates[name] = []
                    gates[name].append(float(val))
    
    if not gates:
        print("No gate values found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gates)))
    
    for (name, values), color in zip(gates.items(), colors):
        # Short name
        parts = name.split('.')
        if 'layer3' in name:
            short = 'Layer3-' + parts[-2].split('[')[-1].rstrip(']')
        elif 'layer4' in name:
            short = 'Layer4-' + parts[-2].split('[')[-1].rstrip(']')
        else:
            short = name[-20:]
        
        ax.plot(epochs[:len(values)], values, linewidth=2, label=short, color=color)
    
    ax.axhline(y=0.119, color='gray', linestyle='--', alpha=0.5, 
               label='Init (sigmoid(-2))')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gate Value (sigmoid)')
    ax.set_title('Learned Interaction Strength During Training')
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 8. Interaction Matrix Heatmap
# =============================================================================

@torch.no_grad()
def plot_interaction_heatmap(model, image_path, save_path="figures/interaction_heatmap.pdf"):
    """Visualize interaction matrices for a sample image."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    from PIL import Image
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    transform = build_eval_transform()
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    
    feats = model.get_features(x)
    
    # Get interaction matrices from first TwistBlock
    from models import TwistBlock
    
    matrices = None
    for block in model.layer3:
        if isinstance(block, TwistBlock):
            matrices = block.mhstci.get_all_interaction_matrices(feats['layer3'])
            break
    
    if matrices is None:
        print("No TwistBlock found")
        return
    
    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 4 direction matrices
    directions = ['0° (H)', '45° (D1)', '90° (V)', '135° (D2)']
    for i, (mat, direction) in enumerate(zip(matrices[:4], directions)):
        mat_np = mat[0].cpu().numpy()
        im = axes[i+1].imshow(mat_np, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[i+1].set_title(f'Direction {direction}')
        axes[i+1].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Correlation')
    plt.suptitle('Spiral-Twisted Channel Interaction Matrices', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--run_dir", type=str, default="runs/main", help="Run directory")
    parser.add_argument("--save_dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--plot", type=str, default="all", 
                        help="Plots to generate: all, bar, radar, scatter, ablation, efficiency")
    parser.add_argument("--dataset", type=str, default="dtd", help="Dataset for single-dataset plots")
    parser.add_argument("--log_file", type=str, help="Training log for gate evolution")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint for feature viz")
    parser.add_argument("--image", type=str, help="Sample image for interaction viz")
    parser.add_argument("--data_dir", type=str, help="Data directory for t-SNE")
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plots = args.plot.lower().split(',')
    
    # Load results if available
    run_dir = Path(args.run_dir)
    aggregated = {}
    if run_dir.exists():
        results = load_results(run_dir)
        if results:
            aggregated = aggregate_results(results)
            print(f"Loaded {len(results)} results from {run_dir}")
    
    # Generate plots
    if 'all' in plots or 'bar' in plots:
        if aggregated:
            plot_bar_chart(aggregated, save_path=str(save_dir / 'bar_chart.pdf'))
    
    if 'all' in plots or 'radar' in plots:
        if aggregated:
            plot_radar_chart(aggregated, save_path=str(save_dir / 'radar_chart.pdf'))
    
    if 'all' in plots or 'scatter' in plots:
        if aggregated:
            plot_params_vs_accuracy(aggregated, dataset=args.dataset,
                                    save_path=str(save_dir / 'params_accuracy.pdf'))
    
    if 'all' in plots or 'ablation' in plots:
        if aggregated:
            plot_ablation_chart(aggregated, dataset=args.dataset,
                               save_path=str(save_dir / 'ablation.pdf'))
    
    if 'all' in plots or 'efficiency' in plots:
        plot_efficiency(save_path=str(save_dir / 'efficiency.pdf'))
    
    if 'all' in plots or 'gate' in plots:
        if args.log_file and Path(args.log_file).exists():
            plot_gate_evolution(args.log_file, save_path=str(save_dir / 'gate_evolution.pdf'))
        else:
            # Try to find a log file
            for log_file in run_dir.glob("*/log.jsonl"):
                if 'twistnet' in str(log_file):
                    plot_gate_evolution(str(log_file), save_path=str(save_dir / 'gate_evolution.pdf'))
                    break
    
    if args.checkpoint and args.image:
        if 'all' in plots or 'interaction' in plots:
            if not Path(args.checkpoint).exists():
                print(f"[Warning] Checkpoint not found: {args.checkpoint}")
            else:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                model = build_model('twistnet18', num_classes=47, pretrained=False)
                model.load_state_dict(state_dict)
                plot_interaction_heatmap(model, args.image, save_path=str(save_dir / 'interaction.pdf'))
    
    if args.checkpoint and args.data_dir:
        if 'all' in plots or 'tsne' in plots:
            if not Path(args.checkpoint).exists():
                print(f"[Warning] Checkpoint not found: {args.checkpoint}")
            else:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                model = build_model('twistnet18', num_classes=47, pretrained=False)
                model.load_state_dict(state_dict)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                
                transform = build_eval_transform()
                _, _, test_loader, _ = get_dataloaders(
                    args.data_dir, args.dataset, fold=1, eval_transform=transform
                )
                plot_tsne(model, test_loader, save_path=str(save_dir / 'tsne.pdf'))
    
    print(f"\nAll figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
