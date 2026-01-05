from models import build_model, count_params 
x = torch.randn(2, 3, 224, 224).cuda() 
models = ['resnet18', 'se_resnet18', 'convnext', 'hornet', 'focalnet', 'van', 'moganet', 'twistnet18'] 
for name in models: 
    model = build_model(name, num_classes=47).cuda() 
    y = model(x) 
    print(f'{name}: {count_params(model)/1e6:.2f}M OK') 
