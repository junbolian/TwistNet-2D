#!/usr/bin/env python3
"""
Visualization tools for TwistNet-2D.

Visualizations:
1. Spiral interaction matrices (channel co-occurrence per direction)
2. Feature maps from different stages
3. Gate value evolution during training
4. Training curves (loss, accuracy)
5. Class-specific co-occurrence patterns
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from models import build_model, TwistBlock
from transforms import build_eval_transform


def load_image(path: str, transform):
    """Load and transform a single image."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def visualize_spiral_interactions(model, image_tensor, save_dir: str = "vis"):
    """Visualize interaction matrices from different spiral directions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)
    
    feats = model.get_features(x)
    
    # Find TwistBlocks and get their interaction matrices
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Spiral-Twisted Interaction Matrices by Direction", fontsize=14)
    
    directions = [r"0° $\rightarrow$", r"45° $\nearrow$", r"90° $\uparrow$", r"135° $\nwarrow$"]
    
    block_idx = 0
    for name, layer in [('layer3', model.layer3), ('layer4', model.layer4)]:
        for block in layer:
            if isinstance(block, TwistBlock):
                # FIXED: use mhstci instead of stci
                matrices = block.mhstci.get_all_interaction_matrices(feats[name])
                for i, mat in enumerate(matrices[:4]):
                    row = block_idx
                    col = i
                    if row < 2 and col < 4:
                        ax = axes[row, col]
                        mat_np = mat[0].cpu().numpy()
                        im = ax.imshow(mat_np, cmap='coolwarm', vmin=-0.5, vmax=0.5)
                        ax.set_title(f"{name} - {directions[i]}")
                        ax.axis('off')
                block_idx += 1
                if block_idx >= 2:
                    break
        if block_idx >= 2:
            break
    
    plt.colorbar(im, ax=axes, shrink=0.6, label='Correlation')
    plt.tight_layout()
    plt.savefig(save_dir / 'spiral_interactions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'spiral_interactions.png'}")


@torch.no_grad()
def visualize_feature_maps(model, image_tensor, save_dir: str = "vis"):
    """Visualize feature maps from different stages."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)
    
    feats = model.get_features(x)
    
    n_feats = len(feats)
    ncols = min(3, n_feats)
    nrows = (n_feats + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    fig.suptitle("Feature Maps at Different Stages", fontsize=14)
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (name, feat) in enumerate(feats.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        feat_np = feat[0].cpu().numpy()
        
        # Show mean activation
        mean_act = feat_np.mean(axis=0)
        ax.imshow(mean_act, cmap='viridis')
        ax.set_title(f'{name} ({feat_np.shape[0]} ch, {feat_np.shape[1]}x{feat_np.shape[2]})')
        ax.axis('off')
    
    # Hide empty axes
    for idx in range(len(feats), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'feature_maps.png'}")


def visualize_gate_evolution(log_file: str, save_dir: str = "vis"):
    """Visualize gate value evolution during training."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    epochs, gates = [], {}
    
    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            if 'gates' in data:
                for name, val in data['gates'].items():
                    if name not in gates:
                        gates[name] = []
                    gates[name].append(float(val) if isinstance(val, str) else val)
    
    if not gates:
        print("No gate values found in log file.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gates)))
    for (name, values), color in zip(gates.items(), colors):
        # Extract short name
        parts = name.split('.')
        short_name = '.'.join(parts[-3:-1]) if len(parts) > 2 else name
        ax.plot(epochs[:len(values)], values, label=short_name, linewidth=2, color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gate Value (sigmoid)', fontsize=12)
    ax.set_title('Gate Value Evolution During Training', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add horizontal line at initial value
    ax.axhline(y=0.119, color='gray', linestyle='--', alpha=0.5, label='init (sigmoid(-2))')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'gate_evolution.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'gate_evolution.png'}")


def visualize_training_curves(log_file: str, save_dir: str = "vis"):
    """Visualize training curves (loss, accuracy)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    epochs, train_loss, val_loss = [], [], []
    train_acc, val_acc, lrs = [], [], []
    
    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            train_loss.append(data['train_loss'])
            val_loss.append(data['val_loss'])
            train_acc.append(data['train_acc'] * 100)  # Convert to percentage
            val_acc.append(data['val_acc'] * 100)
            lrs.append(data.get('lr', 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, label='Train', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_acc, label='Train', linewidth=2)
    axes[1].plot(epochs, val_acc, label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Mark best val accuracy
    best_idx = np.argmax(val_acc)
    axes[1].scatter([epochs[best_idx]], [val_acc[best_idx]], color='red', s=100, zorder=5)
    axes[1].annotate(f'Best: {val_acc[best_idx]:.1f}%', 
                     xy=(epochs[best_idx], val_acc[best_idx]),
                     xytext=(10, -10), textcoords='offset points')
    
    # Learning rate
    axes[2].plot(epochs, lrs, linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'training_curves.png'}")


def compare_runs(run_dirs: list, save_dir: str = "vis"):
    """Compare training curves from multiple runs."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for run_dir, color in zip(run_dirs, colors):
        run_dir = Path(run_dir)
        log_file = run_dir / "log.jsonl"
        
        if not log_file.exists():
            print(f"Log file not found: {log_file}")
            continue
        
        epochs, val_acc, val_loss = [], [], []
        with open(log_file) as f:
            for line in f:
                data = json.loads(line)
                epochs.append(data['epoch'])
                val_acc.append(data['val_acc'] * 100)
                val_loss.append(data['val_loss'])
        
        label = run_dir.name
        axes[0].plot(epochs, val_loss, label=label, linewidth=2, color=color)
        axes[1].plot(epochs, val_acc, label=label, linewidth=2, color=color)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Loss')
    axes[0].set_title('Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'comparison.png'}")


@torch.no_grad()
def visualize_class_patterns(model, dataloader, class_names: list = None, 
                             num_classes: int = 10, save_dir: str = "vis"):
    """Visualize average interaction patterns per class."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    class_interactions = {i: [] for i in range(num_classes)}
    
    for x, y in dataloader:
        x = x.to(device)
        feats = model.get_features(x)
        
        for block in model.layer3:
            if isinstance(block, TwistBlock):
                # FIXED: use mhstci instead of stci
                matrices = block.mhstci.get_all_interaction_matrices(feats['layer3'])
                for i in range(len(y)):
                    if y[i].item() < num_classes:
                        # Average across all heads
                        avg_mat = torch.stack([m[i] for m in matrices]).mean(0)
                        class_interactions[y[i].item()].append(avg_mat.cpu())
                break
        
        # Only process first few batches
        if sum(len(v) for v in class_interactions.values()) > num_classes * 20:
            break
    
    # Plot
    ncols = min(5, num_classes)
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for cls in range(num_classes):
        ax = axes[cls]
        if class_interactions[cls]:
            avg_mat = torch.stack(class_interactions[cls]).mean(0).numpy()
            im = ax.imshow(avg_mat, cmap='coolwarm', vmin=-0.3, vmax=0.3)
            title = class_names[cls] if class_names and cls < len(class_names) else f'Class {cls}'
            ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty axes
    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Average Interaction Patterns per Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'class_patterns.png'}")


@torch.no_grad()
def visualize_direction_selectivity(model, image_paths: list, save_dir: str = "vis"):
    """
    Visualize direction selectivity: different texture orientations activate corresponding direction heads.

    This is the KEY visualization for demonstrating that spiral directions learn orientation-specific patterns.

    Args:
        model: TwistNet model
        image_paths: list of (image_path, label) tuples
            e.g., [('horizontal.jpg', 'Horizontal'), ('vertical.jpg', 'Vertical'), ...]
        save_dir: output directory

    Output layout:
        - Each row = one sample image
        - Column 0 = input image
        - Columns 1-4 = interaction heatmaps for 0°, 45°, 90°, 135°
        - Red border highlights the strongest direction per row
        - μ values shown below each heatmap (red & bold for max)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model.eval()
    device = next(model.parameters()).device
    transform = build_eval_transform()

    n_samples = len(image_paths)
    directions = [r'0° $\rightarrow$', r'45° $\nearrow$', r'90° $\uparrow$', r'135° $\nwarrow$']

    # Professional figure styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })

    # Create figure: N rows × 5 columns (input + 4 directions) + colorbar space
    fig, axes = plt.subplots(n_samples, 5, figsize=(11, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    im_last = None  # For colorbar

    for row, (img_path, label) in enumerate(image_paths):
        # Load image
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # Get features and interaction matrices from layer3
        feats = model.get_features(x)

        matrices = None
        for block in model.layer3:
            if isinstance(block, TwistBlock):
                matrices = block.mhstci.get_all_interaction_matrices(feats['layer3'])
                break

        if matrices is None:
            print(f"[Warning] No TwistBlock found for {img_path}")
            continue

        # === Column 0: Input image ===
        axes[row, 0].imshow(img)
        # Show label below input image (not as ylabel to save space)
        axes[row, 0].set_xlabel(label, fontsize=10, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

        if row == 0:
            axes[row, 0].set_title('Input', fontsize=11, fontweight='bold', pad=8)

        # Compute mean absolute activation for each direction
        mean_acts = [mat[0].abs().mean().item() for mat in matrices[:4]]
        max_dir = np.argmax(mean_acts)

        # === Columns 1-4: Direction heatmaps ===
        for col, (mat, direction) in enumerate(zip(matrices[:4], directions)):
            mat_np = mat[0].cpu().numpy()

            im = axes[row, col+1].imshow(mat_np, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            im_last = im

            # Highlight max direction with red border
            is_max = (col == max_dir)
            if is_max:
                for spine in axes[row, col+1].spines.values():
                    spine.set_edgecolor('#E74C3C')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                for spine in axes[row, col+1].spines.values():
                    spine.set_visible(False)

            # Show μ value below heatmap
            mu = mean_acts[col]
            color = '#E74C3C' if is_max else '#666666'
            weight = 'bold' if is_max else 'normal'
            axes[row, col+1].text(0.5, -0.15, f'$\\mu$={mu:.3f}',
                                   transform=axes[row, col+1].transAxes,
                                   ha='center', fontsize=9, color=color, fontweight=weight)

            axes[row, col+1].set_xticks([])
            axes[row, col+1].set_yticks([])

            # Column titles on first row only
            if row == 0:
                axes[row, col+1].set_title(direction, fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.08, right=0.88)

    # Add colorbar on the far right
    if im_last is not None:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_last, cax=cbar_ax)
        cbar.set_label('Correlation', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Save both PNG and PDF
    plt.savefig(save_dir / 'direction_selectivity.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_dir / 'direction_selectivity.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_dir / 'direction_selectivity.png'}")
    print(f"Saved: {save_dir / 'direction_selectivity.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwistNet Visualization Tools")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--images", type=str, nargs='+',
                        help="Multiple images for direction selectivity: path1:label1 path2:label2 ...")
    parser.add_argument("--log_file", type=str, help="Path to training log (log.jsonl)")
    parser.add_argument("--run_dir", type=str, help="Path to run directory")
    parser.add_argument("--compare", nargs='+', help="Multiple run directories to compare")
    parser.add_argument("--save_dir", type=str, default="vis", help="Output directory")
    parser.add_argument("--num_classes", type=int, default=47)
    args = parser.parse_args()

    transform = build_eval_transform()

    # Direction selectivity visualization (multi-image)
    if args.checkpoint and args.images:
        print("\n[Visualizing direction selectivity]")
        model = build_model("twistnet18", num_classes=args.num_classes, pretrained=False)

        # Handle different checkpoint formats
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        # Parse "path:label" format
        image_info_list = []
        for item in args.images:
            if ':' in item:
                path, label = item.rsplit(':', 1)
            else:
                path = item
                label = Path(item).stem
            image_info_list.append((path, label))

        visualize_direction_selectivity(model, image_info_list, args.save_dir)

    # Single image visualization (spiral interactions + feature maps)
    elif args.checkpoint and args.image:
        print("\n[Visualizing model internals]")
        model = build_model("twistnet18", num_classes=args.num_classes, pretrained=False)

        # Handle different checkpoint formats
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        img_tensor = load_image(args.image, transform)

        visualize_spiral_interactions(model, img_tensor, args.save_dir)
        visualize_feature_maps(model, img_tensor, args.save_dir)
    
    # Visualize training logs
    if args.log_file:
        print("\n[Visualizing training logs]")
        visualize_gate_evolution(args.log_file, args.save_dir)
        visualize_training_curves(args.log_file, args.save_dir)
    
    # Visualize from run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        log_file = run_dir / "log.jsonl"
        if log_file.exists():
            print(f"\n[Visualizing from {run_dir}]")
            visualize_gate_evolution(str(log_file), args.save_dir)
            visualize_training_curves(str(log_file), args.save_dir)
    
    # Compare multiple runs
    if args.compare:
        print("\n[Comparing runs]")
        compare_runs(args.compare, args.save_dir)
    
    print("\nVisualization completed!")
