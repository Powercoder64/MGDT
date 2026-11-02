"""ImageNet-1K in the torchvision ImageFolder layout (``train/<wnid>/*.JPEG``, ``val/<wnid>/*.JPEG``)."""

import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_train_transform(img_size=224, rand_augment="rand-m9-mstd0.5-inc1", random_erasing=0.25,
                             interpolation="bicubic", color_jitter=0.4):
    """RandomResizedCrop, flip, RandAugment and random erasing."""
    from timm.data import create_transform

    return create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=color_jitter,
        auto_augment=rand_augment,
        interpolation=interpolation,
        re_prob=random_erasing,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def imagenet_eval_transform(img_size=224, crop_pct=0.875):
    """Resize to ``img_size / crop_pct`` then a single centre crop."""
    return transforms.Compose([
        transforms.Resize(int(round(img_size / crop_pct)), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_imagenet(root, split, transform):
    """ImageFolder dataset for ``root/split``."""
    path = os.path.join(root, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"ImageNet split directory not found: {path}")
    return datasets.ImageFolder(path, transform=transform)
