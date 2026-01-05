"""
TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition
=========================================================================

Model Groups:
-------------
Group 1 - Fair Comparison (10-15M params, train from scratch):
  - resnet18 (11.7M) - CVPR 2016
  - seresnet18 (11.8M) - CVPR 2018
  - convnextv2_nano (15.6M) - CVPR 2023
  - twistnet18 (11.6M) - Ours

Group 2 - Efficiency Comparison (official large models):
  - convnext_tiny (28M) - CVPR 2022
  - swin_tiny (28M) - ICCV 2021
  - efficientnetv2_s (24M) - ICML 2021

Usage:
------
    from models import build_model, list_models, count_params
    
    # List available models
    list_models()
    
    # Build model
    model = build_model('twistnet18', num_classes=47)
    model = build_model('convnextv2_nano', num_classes=47, pretrained=False)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

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
    # =========================================================================
    # Classic
    'resnet18': {'timm_name': 'resnet18', 'params': '11.7M', 'venue': 'CVPR 2016', 'group': 1},
    'seresnet18': {'timm_name': 'seresnet18', 'params': '11.8M', 'venue': 'CVPR 2018', 'group': 1},
    
    # 2023 Models
    'convnextv2_nano': {'timm_name': 'convnextv2_nano', 'params': '15.6M', 'venue': 'CVPR 2023', 'group': 1},
    'fastvit_sa12': {'timm_name': 'fastvit_sa12', 'params': '10.9M', 'venue': 'ICCV 2023', 'group': 1},
    'efficientformerv2_s1': {'timm_name': 'efficientformerv2_s1', 'params': '12.7M', 'venue': 'ICCV 2023', 'group': 1},
    
    # 2024 Models
    'repvit_m1_5': {'timm_name': 'repvit_m1_5', 'params': '14.0M', 'venue': 'CVPR 2024', 'group': 1},
    
    # Ours
    'twistnet18': {'timm_name': None, 'params': '11.6M', 'venue': 'Ours', 'group': 1},
    
    # =========================================================================
    # Group 2: Efficiency Comparison (official tiny/small ~25-30M)
    # =========================================================================
    'convnext_tiny': {'timm_name': 'convnext_tiny', 'params': '28.6M', 'venue': 'CVPR 2022', 'group': 2},
    'convnextv2_tiny': {'timm_name': 'convnextv2_tiny', 'params': '28.6M', 'venue': 'CVPR 2023', 'group': 2},
    'swin_tiny': {'timm_name': 'swin_tiny_patch4_window7_224', 'params': '28.3M', 'venue': 'ICCV 2021', 'group': 2},
    'maxvit_tiny': {'timm_name': 'maxvit_tiny_tf_224', 'params': '30.9M', 'venue': 'ECCV 2022', 'group': 2},
    
    # =========================================================================
    # Group 3: Additional baselines (various sizes)
    # =========================================================================
    'efficientnet_b0': {'timm_name': 'efficientnet_b0', 'params': '5.3M', 'venue': 'ICML 2019', 'group': 3},
    'efficientnetv2_s': {'timm_name': 'efficientnetv2_rw_s', 'params': '24M', 'venue': 'ICML 2021', 'group': 3},
    'mobilenetv3_large': {'timm_name': 'mobilenetv3_large_100', 'params': '5.4M', 'venue': 'ICCV 2019', 'group': 3},
    'convnextv2_pico': {'timm_name': 'convnextv2_pico', 'params': '9.1M', 'venue': 'CVPR 2023', 'group': 3},
    'regnety_016': {'timm_name': 'regnety_016', 'params': '11.2M', 'venue': 'CVPR 2020', 'group': 3},
}


def list_models():
    """Print available models."""
    print("=" * 75)
    print("Available Models for TwistNet-2D Benchmark")
    print("=" * 75)
    print(f"\n{'Model':<25} {'Params':<10} {'Venue':<15} {'Source'}")
    print("-" * 75)
    
    print("\n[Group 1: Fair Comparison - 10-16M params - MAIN EXPERIMENTS]")
    group1 = ['resnet18', 'seresnet18', 'convnextv2_nano', 'fastvit_sa12', 
              'efficientformerv2_s1', 'repvit_m1_5', 'twistnet18']
    for name in group1:
        info = MODEL_REGISTRY[name]
        source = 'Custom' if info['timm_name'] is None else 'timm'
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15} {source}")
    
    print("\n[Group 2: Efficiency Comparison - Official Tiny Models ~25-30M]")
    group2 = ['convnext_tiny', 'convnextv2_tiny', 'swin_tiny', 'maxvit_tiny']
    for name in group2:
        info = MODEL_REGISTRY[name]
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15} timm")
    
    print("\n[Group 3: Additional Baselines]")
    group3 = ['efficientnet_b0', 'mobilenetv3_large', 'convnextv2_pico', 'regnety_016']
    for name in group3:
        info = MODEL_REGISTRY[name]
        print(f"  {name:<23} {info['params']:<10} {info['venue']:<15} timm")
    
    print("\n" + "=" * 75)


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
    4 directions: 0° (→), 45° (↗), 90° (↑), 135° (↖)
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
            ox, oy = center + dx, center + dy
            if 0 <= ox < k and 0 <= oy < k:
                self.dwconv.weight[:, :, oy, ox] = 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x) * self.scale


class SpiralTwistedInteractionHead(nn.Module):
    """
    Single head of Spiral-Twisted Channel Interaction.
    
    Computes: z_i(x,y) × z_j_twisted(x,y)
    """
    def __init__(self, in_ch: int, c_red: int, direction: int = 0, use_spiral: bool = True):
        super().__init__()
        self.c_red = c_red
        self.use_spiral = use_spiral
        self.direction = direction
        
        self.reduce = nn.Sequential(
            conv1x1(in_ch, c_red),
            nn.BatchNorm2d(c_red),
            nn.ReLU(inplace=True)
        )
        self.spiral = SpiralTwist(c_red, direction) if use_spiral else nn.Identity()
        
        idx_i, idx_j = torch.triu_indices(c_red, c_red, offset=0)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)
        
        self.pair_dim = int(idx_i.numel())
        self.out_dim = c_red + self.pair_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z_twisted = self.spiral(z)
        
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
        z_tw_norm = F.normalize(z_twisted, p=2, dim=1, eps=1e-6)
        
        zi = z_norm.index_select(1, self.idx_i)
        zj = z_tw_norm.index_select(1, self.idx_j)
        pair = zi * zj
        
        return torch.cat([z_norm, pair], dim=1)
    
    def get_interaction_matrix(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z_tw = self.spiral(z)
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
        z_tw_norm = F.normalize(z_tw, p=2, dim=1, eps=1e-6)
        B, C, H, W = z_norm.shape
        z_flat = z_norm.view(B, C, -1)
        z_tw_flat = z_tw_norm.view(B, C, -1)
        return torch.bmm(z_flat, z_tw_flat.transpose(1, 2)) / (H * W)


class AdaptiveInteractionSelection(nn.Module):
    """SE-style attention for selecting important interactions."""
    def __init__(self, ch: int, rd: int = 4):
        super().__init__()
        mid = max(ch // rd, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class MultiHeadSpiralTwistedInteraction(nn.Module):
    """
    Multi-Head Spiral-Twisted Channel Interaction (MH-STCI).
    
    Multiple heads with different spiral directions (0°, 45°, 90°, 135°)
    for rotation-invariant co-occurrence detection.
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
    ):
        super().__init__()
        self.in_ch = base_width
        self.twist_stages = set(twist_stages)
        self.num_heads = num_heads
        self.c_red_list = c_red_list or [8, 8, 8, 8]
        self.use_ais = use_ais
        self.use_spiral = use_spiral
        self.gate_init = gate_init
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
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
# Model Factory
# =============================================================================

def build_model(name: str, num_classes: int = 47, pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Build model by name.
    
    Args:
        name: Model name (see list_models() for options)
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights (for timm models)
        **kwargs: Additional arguments for TwistNet
    
    Returns:
        nn.Module
    
    Examples:
        # TwistNet
        model = build_model('twistnet18', num_classes=47)
        
        # timm models
        model = build_model('resnet18', num_classes=47, pretrained=True)
        model = build_model('convnextv2_nano', num_classes=47, pretrained=False)
    """
    name = name.lower().replace("-", "_")
    
    # TwistNet (custom implementation)
    if name == "twistnet18":
        return TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=kwargs.get("twist_stages", (3, 4)),
            num_heads=kwargs.get("num_heads", 4),
            c_red_list=kwargs.get("c_red_list", [8, 8, 8, 8]),
            use_ais=kwargs.get("use_ais", True),
            use_spiral=kwargs.get("use_spiral", True),
            gate_init=kwargs.get("gate_init", -2.0),
        )
    
    # Ablation: TwistNet without spiral (same position interaction)
    if name == "twistnet18_no_spiral":
        return TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(3, 4),
            num_heads=4,
            use_ais=True,
            use_spiral=False,
        )
    
    # Ablation: TwistNet without AIS
    if name == "twistnet18_no_ais":
        return TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(3, 4),
            num_heads=4,
            use_ais=False,
            use_spiral=True,
        )
    
    # Ablation: TwistNet with only 1st-order (no pairwise products)
    if name == "twistnet18_first_order":
        # This is essentially ResNet-18
        return TwistNet(
            layers=[2, 2, 2, 2],
            num_classes=num_classes,
            twist_stages=(),  # No twist blocks
        )
    
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
    
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


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
            'efficientformerv2_s1', 'repvit_m1_5', 'twistnet18']


def get_efficiency_comparison_models() -> List[str]:
    """Official large models for efficiency comparison (~25-30M)."""
    return ['convnext_tiny', 'convnextv2_tiny', 'swin_tiny', 'maxvit_tiny']


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
    
    # List available models
    list_models()
    
    print("\n" + "=" * 75)
    print("Testing Model Builds")
    print("=" * 75)
    
    x = torch.randn(2, 3, 224, 224)
    
    # Test Group 1: Fair comparison (MAIN)
    print("\n[Group 1: Fair Comparison - 10-16M params - MAIN]")
    for name in get_fair_comparison_models():
        try:
            model = build_model(name, num_classes=47)
            model.eval()
            with torch.no_grad():
                y = model(x)
            params = count_params(model) / 1e6
            print(f"  {name:<25} {params:>6.2f}M  output: {y.shape}  ✓")
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
    
    # Test Group 2: Efficiency comparison (only if timm available)
    if TIMM_AVAILABLE:
        print("\n[Group 2: Efficiency Comparison - Official Tiny Models]")
        for name in get_efficiency_comparison_models():
            try:
                model = build_model(name, num_classes=47, pretrained=False)
                model.eval()
                with torch.no_grad():
                    y = model(x)
                params = count_params(model) / 1e6
                print(f"  {name:<25} {params:>6.2f}M  output: {y.shape}  ✓")
            except Exception as e:
                print(f"  {name:<25} FAILED: {e}")
    
    # Test Ablation models
    print("\n[Ablation Models]")
    for name in get_ablation_models():
        try:
            model = build_model(name, num_classes=47)
            model.eval()
            with torch.no_grad():
                y = model(x)
            params = count_params(model) / 1e6
            print(f"  {name:<30} {params:>6.2f}M  output: {y.shape}  ✓")
        except Exception as e:
            print(f"  {name:<30} FAILED: {e}")
    
    # TwistNet gate values
    print("\n" + "-" * 75)
    print("TwistNet-18 Initial Gate Values:")
    model = build_model("twistnet18", num_classes=47)
    for name, val in list(model.get_gate_values().items())[:4]:
        print(f"  {name.split('.')[-2]}: {val:.4f}")
    
    print("\n" + "=" * 75)
    print("All tests completed!")
    print("=" * 75)