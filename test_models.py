#!/usr/bin/env python3
"""
Quick test script to verify all models build and forward correctly.
"""

import torch
from models import build_model, count_params, list_models

def test_models():
    """Test all models in the registry."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)
    
    print("=" * 70)
    print("Testing TwistNet-2D Models")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Group 1: Fair comparison models (10-16M params)
    print("\n[Group 1: Fair Comparison (10-16M params)]")
    models_group1 = [
        'resnet18', 'seresnet18', 'convnextv2_nano', 
        'fastvit_sa12', 'efficientformerv2_s1', 'repvit_m1_5',
        'twistnet18'
    ]
    
    for name in models_group1:
        try:
            model = build_model(name, num_classes=47, pretrained=False).to(device)
            model.eval()
            with torch.no_grad():
                y = model(x)
            params = count_params(model) / 1e6
            print(f"  {name:<25} {params:>6.2f}M  output: {tuple(y.shape)}  OK")
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
    
    # Ablation models
    print("\n[Ablation Models]")
    ablation_models = ['twistnet18', 'twistnet18_no_spiral', 'twistnet18_no_ais', 'twistnet18_first_order']
    
    for name in ablation_models:
        try:
            model = build_model(name, num_classes=47, pretrained=False).to(device)
            model.eval()
            with torch.no_grad():
                y = model(x)
            params = count_params(model) / 1e6
            print(f"  {name:<25} {params:>6.2f}M  output: {tuple(y.shape)}  OK")
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
    
    # Group 2: Efficiency comparison (larger models)
    print("\n[Group 2: Efficiency Comparison (~25-30M params)]")
    models_group2 = ['convnext_tiny', 'convnextv2_tiny', 'swin_tiny']
    
    for name in models_group2:
        try:
            model = build_model(name, num_classes=47, pretrained=False).to(device)
            model.eval()
            with torch.no_grad():
                y = model(x)
            params = count_params(model) / 1e6
            print(f"  {name:<25} {params:>6.2f}M  output: {tuple(y.shape)}  OK")
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
    
    # Test pretrained loading
    print("\n[Pretrained Weight Loading]")
    try:
        print("  Loading TwistNet-18 with ImageNet pretrained backbone...")
        model = build_model('twistnet18', num_classes=47, pretrained=True).to(device)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"  TwistNet-18 pretrained: OK")
        
        # Check gate values
        gates = model.get_gate_values()
        if gates:
            print(f"  Initial gate values (should be ~0.12):")
            for name, val in list(gates.items())[:2]:
                short_name = name.split('.')[-2] if '.' in name else name
                print(f"    {short_name}: {val:.4f}")
    except Exception as e:
        print(f"  Pretrained loading FAILED: {e}")
    
    # Test ResNet-18 pretrained
    try:
        print("\n  Loading ResNet-18 with ImageNet pretrained weights...")
        model = build_model('resnet18', num_classes=47, pretrained=True).to(device)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"  ResNet-18 pretrained: OK")
    except Exception as e:
        print(f"  ResNet-18 pretrained FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


def quick_test():
    """Quick one-liner test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)
    
    models = ['resnet18', 'seresnet18', 'convnextv2_nano', 'fastvit_sa12', 
              'efficientformerv2_s1', 'repvit_m1_5', 'twistnet18']
    
    for name in models:
        model = build_model(name, num_classes=47, pretrained=False).to(device)
        y = model(x)
        print(f'{name}: {count_params(model)/1e6:.2f}M OK')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_test()
    else:
        test_models()
