"""
TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition
=========================================================================

All models are trained FROM SCRATCH without ImageNet pretraining for fair
architectural comparison. This isolates the contribution of each architecture
from transfer learning effects.

Model Groups:
-------------
Group 1 - Fair Comparison (10-16M params):
  - resnet18 (11.20M) - CVPR 2016
  - seresnet18 (11.29M) - CVPR 2018
  - convnextv2_nano (15.01M) - CVPR 2023
  - fastvit_sa12 (10.60M) - ICCV 2023
  - repvit_m1_5 (13.67M) - CVPR 2024
  - twistnet18 (11.59M) - Ours

Group 2 - Efficiency Comparison (official large models ~28M):
  - convnext_tiny (27.86M) - CVPR 2022
  - swin_tiny (27.56M) - ICCV 2021

Usage:
------
    from models import build_model, list_models, count_params

    # Build model from scratch (RECOMMENDED for fair comparison)
    model = build_model('twistnet18', num_classes=47, pretrained=False)
    model = build_model('resnet18', num_classes=47, pretrained=False)

    # Build model with pretrained weights (for transfer learning experiments)
    model = build_model('resnet18', num_classes=47, pretrained=True)
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if timm is available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Only TwistNet models available.")
    print("Install with: pip install timm")


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    # =========================================================================
    # Group 1: Fair Comparison (10-16M params) - Main experiments
    # All models have ImageNet pretrained weights available
    # =========================================================================
    'resnet18': {'timm_name': 'resnet18', 'params': '11.20M', 'venue': 'CVPR 2016', 'group': 1, 'pretrained': True},
    'seresnet18': {'timm_name': 'seresnet18', 'params': '11.29M', 'venue': 'CVPR 2018', 'group': 1, 'pretrained': True},  # Uses ResNet-18 weights
    'convnextv2_nano': {'timm_name': 'convnextv2_nano', 'params': '15.01M', 'venue': 'CVPR 2023', 'group': 1, 'pretrained': True},
    'fastvit_sa12': {'timm_name': 'fastvit_sa12', 'params': '10.60M', 'venue': 'ICCV 2023', 'group': 1, 'pretrained': True},
    'repvit_m1_5': {'timm_name': 'repvit_m1_5', 'params': '13.67M', 'venue': 'CVPR 2024', 'group': 1, 'pretrained': True},
    'twistnet18': {'timm_name': None, 'params': '11.59M', 'venue': 'Ours', 'group': 1, 'pretrained': True},

    # =========================================================================
    # Group 2: Efficiency Comparison (official tiny/small ~25-30M)
    # =========================================================================
    'convnext_tiny': {'timm_name': 'convnext_tiny', 'params': '27.86M', 'venue': 'CVPR 2022', 'group': 2, 'pretrained': True},
    'swin_tiny': {'timm_name': 'swin_tiny_patch4_window7_224', 'params': '27.56M', 'venue': 'ICCV 2021', 'group': 2, 'pretrained': True},
    
    # =========================================================================
    # Group 3: Additional baselines (various sizes)
    # =========================================================================
    'efficientnet_b0': {'timm_name': 'efficientnet_b0', 'params': '5.3M', 'venue': 'ICML 2019', 'group': 3, 'pretrained': True},
    'efficientnetv2_s': {'timm_name': 'efficientnetv2_rw_s', 'params': '24M', 'venue': 'ICML 2021', 'group': 3, 'pretrained': True},
    'mobilenetv3_large': {'timm_name': 'mobilenetv3_large_100', 'params': '5.4M', 'venue': 'ICCV 2019', 'group': 3, 'pretrained': True},
    'convnextv2_pico': {'timm_name': 'convnextv2_pico', 'params': '9.1M', 'venue': 'CVPR 2023', 'group': 3, 'pretrained': True},
    'regnety_016': {'timm_name': 'regnety_016', 'params': '11.2M', 'venue': 'CVPR 2020', 'group': 3, 'pretrained': True},
}


def list_models():
    """Print available models."""
    print("=" * 75)
    print("Available Models for TwistNet-2D Benchmark")
    print("=" * 75)
    print(f"\n{'Model':<25} {'Params':<10} {'Venue':<15}")
    print("-" * 75)

    print("\n[Group 1: Fair Comparison - 10-16M params - MAIN EXPERIMENTS]")
    group1 = ['resnet18', 'seresnet18', 'convnextv2_nano', 'fastvit_sa12',
              'repvit_m1_5', 'twistnet18']
    for name in group1:
        info = MODEL_REGISTRY[name]
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15}")

    print("\n[Group 2: Efficiency Comparison - Official Tiny Models ~28M]")
    group2 = ['convnext_tiny', 'swin_tiny']
    for name in group2:
        info = MODEL_REGISTRY[name]
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15}")

    print("\n[Group 3: Additional Baselines]")
    group3 = ['efficientnet_b0', 'mobilenetv3_large', 'convnextv2_pico', 'regnety_016']
    for name in group3:
        info = MODEL_REGISTRY[name]
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15}")

    print("\n" + "=" * 75)
    print("Note: All models trained FROM SCRATCH for fair architectural comparison.")
    print("      Pretrained weights available via pretrained=True for transfer learning.")
    print("=" * 75)


# =============================================================================
# Common Utilities
# =============================================================================

def conv3x3(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False, groups=groups)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


# =============================================================================
# TwistNet Components
# =============================================================================

class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SpiralTwist(nn.Module):
    """
    Spatial Twist with directional displacement.
    
    Captures cross-position correlations essential for texture patterns.
    4 directions: 0° (→), 45° (↘), 90° (↓), 135° (↙)
    """
    def __init__(self, dim: int, direction: int = 0, kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.kernel_size = kernel_size
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, 
                                groups=dim, bias=False)
        self._init_directional_kernel()
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))
    
    def _init_directional_kernel(self):
        with torch.no_grad():
            k = self.kernel_size
            center = k // 2
            self.dwconv.weight.zero_()
            self.dwconv.weight[:, :, center, center] = 0.5
            
            angles = [0, 45, 90, 135]
            angle = math.radians(angles[self.direction])
            dx = int(round(math.cos(angle)))
            dy = int(round(math.sin(angle)))
            
            ny, nx = center + dy, center + dx
            if 0 <= ny < k and 0 <= nx < k:
                self.dwconv.weight[:, :, ny, nx] = 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x) * self.scale


class SpiralTwistedInteractionHead(nn.Module):
    """
    Single STCI Head: Channel reduction -> Spiral Twist -> L2 Norm -> Pairwise Products.
    
    Output: [first-order features, second-order interaction features]
    """
    def __init__(self, in_ch: int, c_red: int = 8, direction: int = 0, use_spiral: bool = True):
        super().__init__()
        self.c_red = c_red
        self.use_spiral = use_spiral
        
        # Channel reduction
        self.reduce = nn.Sequential(
            conv1x1(in_ch, c_red),
            nn.BatchNorm2d(c_red),
            nn.ReLU(inplace=True)
        )
        
        # Spiral twist (directional displacement)
        if use_spiral:
            self.twist = SpiralTwist(c_red, direction)
        else:
            self.twist = nn.Identity()
        
        # Pairwise interaction indices (upper triangular including diagonal)
        self.n_pairs = c_red * (c_red + 1) // 2
        idx_i, idx_j = [], []
        for i in range(c_red):
            for j in range(i, c_red):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("idx_i", torch.tensor(idx_i))
        self.register_buffer("idx_j", torch.tensor(idx_j))
        
        # Output dimension: first-order + second-order
        self.out_dim = c_red + self.n_pairs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z_twist = self.twist(z)
        
        # L2 normalize along channel dimension
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
        z_twist_norm = F.normalize(z_twist, p=2, dim=1, eps=1e-6)
        
        # Pairwise products
        z_i = z_norm[:, self.idx_i]
        z_j = z_twist_norm[:, self.idx_j]
        interactions = z_i * z_j
        
        # Concatenate first-order and second-order features
        return torch.cat([z_norm, interactions], dim=1)
    
    def get_interaction_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Return full C_r x C_r interaction matrix for visualization."""
        z = self.reduce(x)
        z_twist = self.twist(z)
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
        z_twist_norm = F.normalize(z_twist, p=2, dim=1, eps=1e-6)
        
        B, C, H, W = z_norm.shape
        z1 = z_norm.view(B, C, 1, H, W)
        z2 = z_twist_norm.view(B, 1, C, H, W)
        return (z1 * z2).mean(dim=(3, 4))


class AdaptiveInteractionSelection(nn.Module):
    """SE-style attention on interaction channels."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class MultiHeadSpiralTwistedInteraction(nn.Module):
    """
    Multi-Head STCI: Aggregates multiple directional heads with AIS.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        c_red_list: List[int] = None,
        use_ais: bool = True,
        use_spiral: bool = True,
    ):
        super().__init__()
        
        if c_red_list is None:
            c_red_list = [8] * num_heads
        
        actual_heads = min(num_heads, 4)
        
        self.heads = nn.ModuleList([
            SpiralTwistedInteractionHead(in_ch, c_red_list[i % len(c_red_list)], 
                                         direction=i % 4, use_spiral=use_spiral)
            for i in range(actual_heads)
        ])
        
        total_dim = sum(h.out_dim for h in self.heads)
        self.ais = AdaptiveInteractionSelection(total_dim) if use_ais else nn.Identity()
        
        num_groups = min(32, total_dim)
        while total_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, total_dim)
        
        self.proj = nn.Sequential(
            conv1x1(total_dim, out_ch),
            nn.BatchNorm2d(out_ch)
        )
        
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter = torch.cat([h(x) for h in self.heads], dim=1)
        inter = self.norm(self.ais(inter))
        return self.proj(inter)
    
    def get_all_interaction_matrices(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [h.get_interaction_matrix(x) for h in self.heads]


class TwistBlock(nn.Module):
    """ResNet BasicBlock with Multi-Head Spiral-Twisted Interaction."""
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample: nn.Module = None,
        num_heads: int = 4,
        c_red_list: List[int] = None,
        use_ais: bool = True,
        use_spiral: bool = True,
        gate_init: float = -2.0,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
        self.mhstci = MultiHeadSpiralTwistedInteraction(
            out_ch, out_ch,
            num_heads=num_heads,
            c_red_list=c_red_list,
            use_ais=use_ais,
            use_spiral=use_spiral,
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(h))
        inter = self.mhstci(h) * torch.sigmoid(self.gate)
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(main + inter + identity)

    def get_gate_value(self) -> float:
        return torch.sigmoid(self.gate).item()


class TwistNet(nn.Module):
    """
    TwistNet: ResNet-like backbone with Spiral-Twisted Channel Interactions.
    
    Default config (TwistNet-18): ~11.6M params
    """
    def __init__(
        self,
        layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 47,
        base_width: int = 64,
        twist_stages: Tuple[int, ...] = (3, 4),
        num_heads: int = 4,
        c_red_list: List[int] = None,
        use_ais: bool = True,
        use_spiral: bool = True,
        gate_init: float = -2.0,
        stem_type: str = "resnet",  # "resnet" or "lightweight"
    ):
        super().__init__()
        self.in_ch = base_width
        self.twist_stages = set(twist_stages)
        self.num_heads = num_heads
        self.c_red_list = c_red_list or [8, 8, 8, 8]
        self.use_ais = use_ais
        self.use_spiral = use_spiral
        self.gate_init = gate_init
        
        # Stem options:
        # - "resnet": 7×7 conv + maxpool, output 56×56 (same FLOPs as ResNet)
        # - "lightweight": 3×3 conv stride 4, output 56×56 (fewer params, same FLOPs)
        if stem_type == "resnet":
            self.stem = nn.Sequential(
                nn.Conv2d(3, base_width, 7, 2, 3, bias=False),  # 224 -> 112
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112 -> 56
            )
        elif stem_type == "lightweight":
            # Two 3×3 convs to achieve 4× downsampling (same output size as resnet stem)
            self.stem = nn.Sequential(
                nn.Conv2d(3, base_width // 2, 3, 2, 1, bias=False),  # 224 -> 112
                nn.BatchNorm2d(base_width // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width // 2, base_width, 3, 2, 1, bias=False),  # 112 -> 56
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}")
        
        widths = [base_width * (2 ** i) for i in range(4)]
        self.layer1 = self._make_layer(1, widths[0], layers[0], 1)
        self.layer2 = self._make_layer(2, widths[1], layers[1], 2)
        self.layer3 = self._make_layer(3, widths[2], layers[2], 2)
        self.layer4 = self._make_layer(4, widths[3], layers[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, stage_id: int, out_ch: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                conv1x1(self.in_ch, out_ch, stride),
                nn.BatchNorm2d(out_ch)
            )
        
        use_twist = (stage_id in self.twist_stages)
        layers = []
        
        if use_twist:
            layers.append(TwistBlock(
                self.in_ch, out_ch, stride, downsample,
                self.num_heads, self.c_red_list, self.use_ais, self.use_spiral, self.gate_init
            ))
        else:
            layers.append(BasicBlock(self.in_ch, out_ch, stride, downsample))
        
        self.in_ch = out_ch
        
        for _ in range(1, blocks):
            if use_twist:
                layers.append(TwistBlock(
                    self.in_ch, out_ch, 1, None,
                    self.num_heads, self.c_red_list, self.use_ais, self.use_spiral, self.gate_init
                ))
            else:
                layers.append(BasicBlock(self.in_ch, out_ch, 1, None))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

    def get_gate_values(self) -> Dict[str, float]:
        return {
            name: m.get_gate_value()
            for name, m in self.named_modules()
            if isinstance(m, TwistBlock)
        }
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stem(x)
        features['stem'] = x
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features


# =============================================================================
# Pretrained Weight Loading
# =============================================================================

def load_pretrained_resnet18_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Load ResNet-18 ImageNet pretrained weights into TwistNet.
    
    CRITICAL: Only loads stem + layer1 + layer2.
    Layer3 and layer4 (with STCI) are kept randomly initialized.
    
    Reason: ResNet's layer3/4 weights are optimized for direct classification,
    but TwistNet's layer3/4 need to work WITH STCI modules. Loading pretrained
    weights for these layers actually HURTS performance because the pretrained
    conv expects its output to be used directly, not mixed with STCI output.
    """
    if not TIMM_AVAILABLE:
        print("[Warning] timm not available, cannot load pretrained weights")
        return model
    
    # Load pretrained ResNet-18
    resnet18 = timm.create_model('resnet18', pretrained=True)
    resnet_state = resnet18.state_dict()
    model_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for key in model_state.keys():
        # CRITICAL: Skip layer3 and layer4 entirely
        # These layers have STCI and need fresh training
        if key.startswith('layer3') or key.startswith('layer4'):
            skipped_keys.append(f"{key} (STCI layer)")
            continue
        
        # Skip classifier (different num_classes)
        if 'fc' in key:
            skipped_keys.append(key)
            continue
        
        # Map stem
        if key.startswith('stem.0'):  # conv
            resnet_key = key.replace('stem.0', 'conv1')
        elif key.startswith('stem.1'):  # bn
            resnet_key = key.replace('stem.1', 'bn1')
        else:
            resnet_key = key
        
        # Check if exists in ResNet weights
        if resnet_key in resnet_state:
            if model_state[key].shape == resnet_state[resnet_key].shape:
                model_state[key] = resnet_state[resnet_key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch)")
        else:
            skipped_keys.append(f"{key} (not in resnet)")
    
    model.load_state_dict(model_state)
    
    if verbose:
        print(f"[Pretrained] Loaded {len(loaded_keys)} layers (stem + layer1 + layer2)")
        print(f"[Pretrained] Layer3/4: random init (STCI needs fresh training)")
    
    return model


def load_twistnet_pretrained_weights(model: nn.Module, weights_path: str, verbose: bool = True) -> nn.Module:
    """
    Load TwistNet-specific ImageNet pretrained weights.
    
    This loads weights from a TwistNet that was pretrained on ImageNet,
    so ALL layers (including STCI) are properly initialized.
    
    Args:
        model: TwistNet model
        weights_path: Path to twistnet18_imagenet.pt
        verbose: Print loading info
    
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Handle DDP wrapped state dict
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key in model_state.keys():
        # Skip classifier (different num_classes)
        if 'fc' in key:
            skipped_keys.append(key)
            continue
        
        if key in state_dict:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch)")
        else:
            skipped_keys.append(f"{key} (not in checkpoint)")
    
    model.load_state_dict(model_state)

    if verbose:
        print(f"[TwistNet Pretrained] Loaded {len(loaded_keys)} layers from {weights_path}")
        print(f"[TwistNet Pretrained] Skipped {len(skipped_keys)} layers (FC/incompatible)")

    return model


# =============================================================================
# Models without official pretrained weights (need manual weight loading)
# =============================================================================

MODELS_WITHOUT_PRETRAINED = {
    'seresnet18',  # No official ImageNet weights in timm
}


def load_pretrained_resnet18_to_seresnet18(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Load ResNet-18 ImageNet pretrained weights into SE-ResNet-18.
    Only loads compatible layers (conv, bn). SE modules remain randomly initialized.
    """
    if not TIMM_AVAILABLE:
        print("[Warning] timm not available, cannot load pretrained weights")
        return model
    
    # Load pretrained ResNet-18
    resnet18 = timm.create_model('resnet18', pretrained=True)
    resnet_state = resnet18.state_dict()
    model_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for key in model_state.keys():
        # Skip SE modules
        if 'se_module' in key or 'se.' in key:
            skipped_keys.append(key)
            continue
        
        # Skip classifier (different num_classes)
        if 'fc' in key:
            skipped_keys.append(key)
            continue
        
        # Check if exists in ResNet weights with same shape
        if key in resnet_state:
            if model_state[key].shape == resnet_state[key].shape:
                model_state[key] = resnet_state[key]
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch)")
        else:
            skipped_keys.append(f"{key} (not in resnet)")
    
    model.load_state_dict(model_state)
    
    if verbose:
        print(f"[Pretrained] Loaded {len(loaded_keys)} layers from ResNet-18 into SE-ResNet-18")
        print(f"[Pretrained] Skipped {len(skipped_keys)} layers (SE modules/FC/incompatible)")
    
    return model


# =============================================================================
# Model Factory
# =============================================================================

def _load_twistnet_pretrained(model: nn.Module, pretrained) -> nn.Module:
    """
    Helper function to load TwistNet pretrained weights.
    
    Auto-detection order:
    1. If pretrained is a path string, use that path
    2. Check ./weights/twistnet18_imagenet.pt
    3. Check <script_dir>/weights/twistnet18_imagenet.pt
    4. Fallback to ResNet-18 partial weights (not recommended)
    """
    # Case 1: Explicit path provided
    if isinstance(pretrained, str) and os.path.exists(pretrained):
        return load_twistnet_pretrained_weights(model, pretrained)
    
    # Case 2: Auto-detect weights file
    if pretrained is True:
        # Search paths in order
        search_paths = [
            "weights/twistnet18_imagenet.pt",  # Current working directory
            os.path.join(os.path.dirname(__file__), "weights", "twistnet18_imagenet.pt"),  # Script directory
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"[Auto-detected] Found TwistNet weights: {path}")
                return load_twistnet_pretrained_weights(model, path)
        
        # Fallback: ResNet-18 partial weights
        print("[Warning] TwistNet-specific weights not found at:")
        for path in search_paths:
            print(f"  - {path}")
        print("[Warning] Falling back to ResNet-18 partial weights (not recommended)")
        print("[Warning] Run pretrain_imagenet.py first for best results!")
        return load_pretrained_resnet18_weights(model)
    
    return model


def build_model(name: str, num_classes: int = 47, pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Build model by name.

    Args:
        name: Model name (see list_models() for options)
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights (default: False for fair comparison)
        **kwargs: Additional arguments for TwistNet

    Returns:
        nn.Module

    Examples:
        # From scratch (RECOMMENDED for fair architectural comparison)
        model = build_model('twistnet18', num_classes=47, pretrained=False)
        model = build_model('resnet18', num_classes=47, pretrained=False)

        # With pretrained (for transfer learning experiments)
        model = build_model('resnet18', num_classes=47, pretrained=True)
    """
    name = name.lower().replace("-", "_")
    
    # TwistNet (custom implementation)
    if name == "twistnet18":
        model = TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=kwargs.get("twist_stages", (3, 4)),
            num_heads=kwargs.get("num_heads", 4),
            c_red_list=kwargs.get("c_red_list", [8, 8, 8, 8]),
            use_ais=kwargs.get("use_ais", True),
            use_spiral=kwargs.get("use_spiral", True),
            gate_init=kwargs.get("gate_init", -2.0),
            stem_type=kwargs.get("stem_type", "resnet"),  # Use ResNet stem for fair FLOPs comparison
        )
        if pretrained:
            model = _load_twistnet_pretrained(model, pretrained)
        return model
    
    # Ablation: TwistNet without spiral (same position interaction)
    if name == "twistnet18_no_spiral":
        model = TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(3, 4),
            num_heads=4,
            use_ais=True,
            use_spiral=False,
            stem_type="resnet",
        )
        if pretrained:
            model = _load_twistnet_pretrained(model, pretrained)
        return model
    
    # Ablation: TwistNet without AIS
    if name == "twistnet18_no_ais":
        model = TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(3, 4),
            num_heads=4,
            use_ais=False,
            use_spiral=True,
            stem_type="resnet",
        )
        if pretrained:
            model = _load_twistnet_pretrained(model, pretrained)
        return model
    
    # Ablation: TwistNet with only 1st-order (no pairwise products)
    if name == "twistnet18_first_order":
        model = TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(),  # No twist blocks
            stem_type="resnet",
        )
        if pretrained:
            model = _load_twistnet_pretrained(model, pretrained)
        return model
    
    # timm models
    if not TIMM_AVAILABLE:
        raise RuntimeError(f"timm not installed. Cannot build '{name}'. Install with: pip install timm")
    
    if name not in MODEL_REGISTRY:
        # Try direct timm lookup
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        except:
            raise ValueError(f"Unknown model: {name}. Use list_models() to see available options.")
    
    timm_name = MODEL_REGISTRY[name]['timm_name']
    if timm_name is None:
        raise ValueError(f"Model '{name}' is not a timm model.")
    
    # Handle models without official pretrained weights
    if name in MODELS_WITHOUT_PRETRAINED and pretrained:
        print(f"[Info] {name} has no official pretrained weights in timm.")
        print(f"[Info] Loading ResNet-18 weights into compatible layers...")
        # Create model without pretrained
        model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        # Load ResNet-18 weights into compatible layers
        if name == 'seresnet18':
            model = load_pretrained_resnet18_to_seresnet18(model)
        return model
    
    # Try to load with pretrained, fallback to scratch if fails
    try:
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    except RuntimeError as e:
        if "No pretrained weights" in str(e) and pretrained:
            print(f"[Warning] {name} has no pretrained weights available. Training from scratch.")
            return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        raise


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Convenience Functions
# =============================================================================

def get_fair_comparison_models() -> List[str]:
    """Models for fair comparison (10-16M params) - MAIN EXPERIMENTS."""
    return ['resnet18', 'seresnet18', 'convnextv2_nano', 'fastvit_sa12',
            'repvit_m1_5', 'twistnet18']


def get_efficiency_comparison_models() -> List[str]:
    """Official large models for efficiency comparison (~28M)."""
    return ['convnext_tiny', 'swin_tiny']


def get_ablation_models() -> List[str]:
    """Models for ablation study."""
    return ['twistnet18', 'twistnet18_no_spiral', 'twistnet18_no_ais', 'twistnet18_first_order']


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("TwistNet-2D Model Zoo")
    print("=" * 75)

    list_models()

    print("\n" + "=" * 75)
    print("Testing Model Builds (from scratch)")
    print("=" * 75)

    x = torch.randn(2, 3, 224, 224)

    # Test TwistNet from scratch
    print("\n[TwistNet-18 (from scratch)]")
    model = build_model('twistnet18', num_classes=47, pretrained=False)
    model.eval()
    with torch.no_grad():
        y = model(x)
    params = count_params(model) / 1e6
    print(f"  twistnet18: {params:.2f}M  output: {y.shape}")

    # Test gate values
    print("\nInitial gate values (sigmoid of gate_init=-2.0):")
    for name, val in list(model.get_gate_values().items())[:4]:
        print(f"  {name.split('.')[-2]}: {val:.4f}")

    print("\n" + "=" * 75)
    print("All tests completed!")
    print("=" * 75)
