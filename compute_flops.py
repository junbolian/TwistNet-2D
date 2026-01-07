#!/usr/bin/env python3
"""
compute_flops.py - Compute FLOPs and parameters for all models

=== INSTALLATION ===
pip install fvcore   # Recommended (Facebook, most accurate)
# OR
pip install thop     # Alternative (PyTorch-OpCounter)
# OR  
pip install ptflops  # Alternative

=== USAGE ===
python compute_flops.py                    # All models
python compute_flops.py --model twistnet18 # Specific model
python compute_flops.py --latex            # LaTeX table output

=== REFERENCE VALUES (from timm/papers) ===
Model                   Params      FLOPs (224x224)
-----------------------------------------------------
resnet18                11.69M      1.82G
seresnet18              11.78M      1.82G
convnextv2_nano         15.62M      2.45G
fastvit_sa12            10.93M      1.42G
efficientformerv2_s1    12.70M      0.66G
repvit_m1_5             14.04M      2.30G
convnext_tiny           28.59M      4.47G
swin_tiny               28.29M      4.51G

=== MANUAL FLOPs CALCULATION FOR TWISTNET ===

ResNet-18 Base FLOPs: 1.82G

Additional FLOPs from each TwistBlock (4 blocks in Stage 3-4):
1. Channel Reduction (1x1 conv): C_in × C_r × H × W
   Stage 3: 256 × 8 × 28 × 28 = 1.6M per block
   Stage 4: 512 × 8 × 14 × 14 = 0.8M per block
   
2. Spiral DWConv (3x3 depthwise): C_r × 9 × H × W
   Stage 3: 8 × 9 × 28 × 28 = 0.06M per block
   Stage 4: 8 × 9 × 14 × 14 = 0.01M per block

3. Pairwise Products: P × H × W (P=36 pairs, 4 heads)
   Stage 3: 36 × 4 × 28 × 28 = 0.11M per block
   Stage 4: 36 × 4 × 14 × 14 = 0.03M per block

4. AIS (2 FC layers): 2 × D × D_mid
   D = 176, D_mid = 44
   2 × 176 × 44 = 0.015M per block

5. Projection (1x1 conv): D × C_out × H × W
   Stage 3: 176 × 256 × 28 × 28 = 35.3M per block
   Stage 4: 176 × 512 × 14 × 14 = 17.6M per block

Total per TwistBlock:
   Stage 3: ~37M FLOPs × 2 blocks = 74M
   Stage 4: ~18.5M FLOPs × 2 blocks = 37M
   
Total STCI overhead: ~111M = 0.11G
TwistNet-18 estimated: 1.82G + 0.11G ≈ 1.93G

FLOPs overhead: 0.11G / 1.82G ≈ 6% (not 8%!)

Note: The 8% in the paper might include other overheads or 
use different counting conventions. Verify with actual measurement.
"""

import argparse
import sys

# Try different FLOPs calculation libraries
FLOPS_BACKEND = None

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FLOPS_BACKEND = 'fvcore'
except ImportError:
    pass

if FLOPS_BACKEND is None:
    try:
        from thop import profile, clever_format
        FLOPS_BACKEND = 'thop'
    except ImportError:
        pass

if FLOPS_BACKEND is None:
    try:
        from ptflops import get_model_complexity_info
        FLOPS_BACKEND = 'ptflops'
    except ImportError:
        pass

import torch
from models import build_model, list_models, count_params


def compute_flops_fvcore(model, input_size=(1, 3, 224, 224)):
    """Compute FLOPs using fvcore (Facebook's library, most accurate)."""
    model.eval()
    x = torch.randn(input_size)
    flops = FlopCountAnalysis(model, x)
    return flops.total()


def compute_flops_thop(model, input_size=(1, 3, 224, 224)):
    """Compute FLOPs using thop."""
    model.eval()
    x = torch.randn(input_size)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return flops


def compute_flops_ptflops(model, input_size=(3, 224, 224)):
    """Compute FLOPs using ptflops."""
    flops, params = get_model_complexity_info(
        model, input_size, as_strings=False, print_per_layer_stat=False
    )
    return flops


def compute_flops(model, input_size=(1, 3, 224, 224)):
    """Compute FLOPs using available backend."""
    if FLOPS_BACKEND == 'fvcore':
        return compute_flops_fvcore(model, input_size)
    elif FLOPS_BACKEND == 'thop':
        return compute_flops_thop(model, input_size)
    elif FLOPS_BACKEND == 'ptflops':
        return compute_flops_ptflops(model, input_size[1:])
    else:
        return None


def format_flops(flops):
    """Format FLOPs as human-readable string."""
    if flops is None:
        return "N/A"
    if flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    else:
        return f"{flops/1e3:.2f}K"


def format_params(params):
    """Format parameters as human-readable string."""
    if params >= 1e6:
        return f"{params/1e6:.2f}M"
    elif params >= 1e3:
        return f"{params/1e3:.2f}K"
    else:
        return str(params)


def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs and parameters")
    parser.add_argument("--model", type=str, default=None, help="Specific model to compute")
    parser.add_argument("--latex", action="store_true", help="Output as LaTeX table")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_classes", type=int, default=47, help="Number of classes (for classifier)")
    args = parser.parse_args()
    
    print(f"FLOPs Backend: {FLOPS_BACKEND or 'None (install fvcore, thop, or ptflops)'}")
    print(f"Input size: {args.img_size}x{args.img_size}")
    print(f"Num classes: {args.num_classes}")
    print()
    
    if FLOPS_BACKEND is None:
        print("ERROR: No FLOPs calculation library found!")
        print("Install one of:")
        print("  pip install fvcore      # Recommended (Facebook)")
        print("  pip install thop        # Alternative")
        print("  pip install ptflops     # Alternative")
        print()
        print("Showing reference values from timm/papers instead:")
        print("="*60)
        reference = [
            ("resnet18", "11.69M", "1.82G"),
            ("seresnet18", "11.78M", "1.82G"),
            ("convnextv2_nano", "15.62M", "2.45G"),
            ("fastvit_sa12", "10.93M", "1.42G"),
            ("efficientformerv2_s1", "12.70M", "0.66G"),
            ("repvit_m1_5", "14.04M", "2.30G"),
            ("twistnet18", "~11.6M", "~1.93G (estimated)"),
            ("convnext_tiny", "28.59M", "4.47G"),
            ("swin_tiny", "28.29M", "4.51G"),
        ]
        for name, params, flops in reference:
            print(f"{name:30s} | Params: {params:>10s} | FLOPs: {flops:>15s}")
        sys.exit(0)
    
    # Models to compute
    if args.model:
        models_to_compute = [args.model]
    else:
        # All comparison models
        models_to_compute = [
            # Group 1: Parameter-matched
            'resnet18',
            'seresnet18', 
            'convnextv2_nano',
            'fastvit_sa12',
            'efficientformerv2_s1',
            'repvit_m1_5',
            'twistnet18',
            # Group 2: Larger
            'convnext_tiny',
            'swin_tiny',
            # Ablation
            'twistnet18_no_spiral',
            'twistnet18_no_ais',
            'twistnet18_first_order',
        ]
    
    results = []
    
    for model_name in models_to_compute:
        try:
            model = build_model(model_name, num_classes=args.num_classes, pretrained=False)
            model.eval()
            
            params = count_params(model)
            flops = compute_flops(model, input_size=(1, 3, args.img_size, args.img_size))
            
            results.append({
                'name': model_name,
                'params': params,
                'params_str': format_params(params),
                'flops': flops,
                'flops_str': format_flops(flops),
            })
            
            print(f"{model_name:30s} | Params: {format_params(params):>10s} | FLOPs: {format_flops(flops):>10s}")
            
        except Exception as e:
            print(f"{model_name:30s} | ERROR: {e}")
            results.append({
                'name': model_name,
                'params': None,
                'params_str': 'ERROR',
                'flops': None,
                'flops_str': 'ERROR',
            })
    
    # Compute overhead for TwistNet vs ResNet-18
    print("\n" + "="*70)
    print("OVERHEAD CALCULATION")
    print("="*70)
    
    resnet_result = next((r for r in results if r['name'] == 'resnet18'), None)
    twistnet_result = next((r for r in results if r['name'] == 'twistnet18'), None)
    
    if resnet_result and twistnet_result and resnet_result['params'] and twistnet_result['params']:
        param_diff = twistnet_result['params'] - resnet_result['params']
        param_overhead = param_diff / resnet_result['params'] * 100
        print(f"ResNet-18 Params:    {format_params(resnet_result['params'])}")
        print(f"TwistNet-18 Params:  {format_params(twistnet_result['params'])}")
        print(f"Difference:          {format_params(param_diff)}")
        print(f"Parameter Overhead:  {param_overhead:.1f}%")
        print()
        
        if resnet_result['flops'] and twistnet_result['flops']:
            flops_diff = twistnet_result['flops'] - resnet_result['flops']
            flops_overhead = flops_diff / resnet_result['flops'] * 100
            print(f"ResNet-18 FLOPs:     {format_flops(resnet_result['flops'])}")
            print(f"TwistNet-18 FLOPs:   {format_flops(twistnet_result['flops'])}")
            print(f"Difference:          {format_flops(flops_diff)}")
            print(f"FLOPs Overhead:      {flops_overhead:.1f}%")
    
    # LaTeX output
    if args.latex:
        print("\n" + "="*70)
        print("LaTeX Table:")
        print("="*70)
        print(r"\begin{tabular}{l c c}")
        print(r"\toprule")
        print(r"\textbf{Model} & \textbf{Params} & \textbf{FLOPs} \\")
        print(r"\midrule")
        for r in results:
            name_escaped = r['name'].replace('_', r'\_')
            print(f"{name_escaped} & {r['params_str']} & {r['flops_str']} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
