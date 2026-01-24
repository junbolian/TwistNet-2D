"""
Data augmentation transforms for TwistNet-2D.

Optimized for training from scratch on texture and fine-grained datasets.
Uses unified crop scale (0.2, 1.0) which better preserves texture structure
compared to ImageNet default (0.08, 1.0).

Augmentation Pipeline:
- RandomResizedCrop: scale=(0.2, 1.0), BICUBIC interpolation
- RandomHorizontalFlip: p=0.5
- RandAugment: n=2 operations, magnitude=9
- ImageNet normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
"""

from torchvision import transforms
from torchvision.transforms import autoaugment

# ImageNet normalization (standard practice)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224, use_randaugment: bool = True, 
                          ra_n: int = 2, ra_m: int = 9, crop_scale: tuple = (0.2, 1.0)):
    """
    Build training transform for texture recognition.
    
    Args:
        img_size: Target image size
        use_randaugment: Whether to use RandAugment
        ra_n: Number of augmentation operations (default 2)
        ra_m: Magnitude of augmentation (default 9)
        crop_scale: RandomResizedCrop scale range (default (0.2, 1.0))
                   - (0.08, 1.0) is ImageNet default
                   - (0.2, 1.0) better preserves texture structure
    
    Returns:
        Composed transform
    """
    t = [
        transforms.RandomResizedCrop(img_size, scale=crop_scale, 
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    
    if use_randaugment and ra_n > 0:
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


# =============================================================================
# Dataset-specific Transform Presets
# =============================================================================

def get_dataset_transform_config(dataset: str) -> dict:
    """
    Get recommended transform configuration for each dataset.
    
    All datasets use unified crop_scale=(0.2, 1.0) for:
    - Fair comparison across datasets
    - Better texture preservation than ImageNet default (0.08, 1.0)
    - Sufficient augmentation for small datasets
    """
    # Unified configuration for all datasets
    default_config = {
        "crop_scale": (0.2, 1.0),  # Unified: preserves texture while allowing augmentation
        "ra_n": 2,
        "ra_m": 9,
    }
    return default_config


def build_train_transform_for_dataset(dataset: str, img_size: int = 224):
    """Build optimized training transform for a specific dataset."""
    config = get_dataset_transform_config(dataset)
    return build_train_transform(
        img_size=img_size,
        use_randaugment=True,
        ra_n=config["ra_n"],
        ra_m=config["ra_m"],
        crop_scale=config["crop_scale"]
    )