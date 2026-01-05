#!/usr/bin/env python3
"""
Visualization tools for TwistNet-2D.

Visualizations:
1. Spiral interaction matrices (channel co-occurrence per direction)
2. Feature maps from different stages
3. Gate value evolution during training
4. Attention/importance from AIS
5. Class-specific co-occurrence patterns
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models import build_model, TwistBlock
from transforms import build_eval_transform


def load_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def visualize_spiral_interactions(model, image_tensor, save_dir: str = "vis"):
    """Visualize interaction matrices from different spiral directions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)
    
    feats = model.get_features(x)
    
    # Find TwistBlocks and get their interaction matrices
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Spiral-Twisted Interaction Matrices by Direction", fontsize=14)
    
    directions = ["0° (horizontal)", "45° (diagonal)", "90° (vertical)", "135° (anti-diag)"]
    
    block_idx = 0
    for name, layer in [('layer3', model.layer3), ('layer4', model.layer4)]:
        for block in layer:
            if isinstance(block, TwistBlock):
                matrices = block.stci.get_all_interaction_matrices(feats[name])
                for i, mat in enumerate(matrices[:4]):
                    row = block_idx
                    col = i
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
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    x = image_tensor.to(device)
    
    feats = model.get_features(x)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Feature Maps at Different Stages", fontsize=14)
    
    for idx, (name, feat) in enumerate(list(feats.items())[:6]):
        ax = axes[idx // 3, idx % 3]
        feat_np = feat[0].cpu().numpy()
        
        # Show mean activation
        mean_act = feat_np.mean(axis=0)
        ax.imshow(mean_act, cmap='viridis')
        ax.set_title(f'{name} (mean of {feat_np.shape[0]} channels)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'feature_maps.png'}")


def visualize_gate_evolution(log_file: str, save_dir: str = "vis"):
    """Visualize gate value evolution during training."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
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
        print("No gate values found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(gates)))
    for (name, values), color in zip(gates.items(), colors):
        short_name = name.split('.')[-2] if '.' in name else name
        ax.plot(epochs[:len(values)], values, label=short_name, linewidth=2, color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gate Value (sigmoid)', fontsize=12)
    ax.set_title('Gate Value Evolution During Training', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'gate_evolution.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'gate_evolution.png'}")


def visualize_training_curves(log_file: str, save_dir: str = "vis"):
    """Visualize training curves (loss, accuracy)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs, train_loss, val_loss = [], [], []
    train_acc, val_acc = [], []
    
    with open(log_file) as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            train_loss.append(data['train_loss'])
            val_loss.append(data['val_loss'])
            train_acc.append(data['train_acc'])
            val_acc.append(data['val_acc'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'training_curves.png'}")


@torch.no_grad()
def visualize_class_patterns(model, dataloader, num_classes: int = 10, save_dir: str = "vis"):
    """Visualize average interaction patterns per class."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    class_interactions = {i: [] for i in range(num_classes)}
    
    for x, y in dataloader:
        x = x.to(device)
        feats = model.get_features(x)
        
        for block in model.layer3:
            if isinstance(block, TwistBlock):
                matrices = block.stci.get_all_interaction_matrices(feats['layer3'])
                for i in range(len(y)):
                    if y[i].item() < num_classes:
                        # Average across all heads
                        avg_mat = torch.stack([m[i] for m in matrices]).mean(0)
                        class_interactions[y[i].item()].append(avg_mat.cpu())
                break
    
    # Plot
    ncols = min(5, num_classes)
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if num_classes > 1 else [axes]
    
    for cls in range(num_classes):
        ax = axes[cls]
        if class_interactions[cls]:
            avg_mat = torch.stack(class_interactions[cls]).mean(0).numpy()
            im = ax.imshow(avg_mat, cmap='coolwarm', vmin=-0.3, vmax=0.3)
            ax.set_title(f'Class {cls}')
        ax.axis('off')
    
    plt.suptitle('Average Interaction Patterns per Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'class_patterns.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--log_file", type=str, help="Path to training log")
    parser.add_argument("--save_dir", type=str, default="vis")
    parser.add_argument("--num_classes", type=int, default=47)
    args = parser.parse_args()
    
    transform = build_eval_transform()
    
    if args.checkpoint and args.image:
        model = build_model("twistnet18", num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        model.eval()
        
        img_tensor = load_image(args.image, transform)
        
        visualize_spiral_interactions(model, img_tensor, args.save_dir)
        visualize_feature_maps(model, img_tensor, args.save_dir)
    
    if args.log_file:
        visualize_gate_evolution(args.log_file, args.save_dir)
        visualize_training_curves(args.log_file, args.save_dir)
    
    print("\nVisualization completed!")
