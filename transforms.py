"""Data augmentation transforms for TwistNet-2D."""

from torchvision import transforms
from torchvision.transforms import autoaugment

# ImageNet normalization (CRITICAL for pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224, use_randaugment: bool = True, 
                          ra_n: int = 2, ra_m: int = 9):
    """
    Build training transform with RandAugment.
    
    Args:
        img_size: Target image size
        use_randaugment: Whether to use RandAugment
        ra_n: Number of augmentation operations
        ra_m: Magnitude of augmentation
    
    Returns:
        Composed transform
    """
    t = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), 
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        t.append(autoaugment.RandAugment(num_ops=ra_n, magnitude=ra_m))
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return transforms.Compose(t)


def build_eval_transform(img_size: int = 224, crop_pct: float = 0.875):
    """
    Build evaluation transform (resize + center crop).
    
    Args:
        img_size: Target image size
        crop_pct: Center crop percentage
    
    Returns:
        Composed transform
    """
    resize = int(img_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
