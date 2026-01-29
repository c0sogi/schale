"""Augmentation pipeline for icon classification training."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform() -> A.Compose:
    """
    Get training augmentation pipeline with strong augmentations.

    Returns:
        Albumentations compose object with training transforms.
    """
    transforms = [
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.5,
        ),
        # Blur and noise
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.3,
        ),
        A.GaussNoise(
            std_range=(0.01, 0.05),
            mean_range=(0.0, 0.0),
            p=0.3,
        ),
        # Geometric augmentations
        A.Rotate(
            limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomScale(
            scale_limit=0.1,
            p=0.5,
        ),
        # Resize to fixed size (required for batching)
        A.Resize(height=116, width=146),
        # Normalization (ImageNet stats)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transform() -> A.Compose:
    """
    Get validation transform pipeline (minimal augmentation).

    Returns:
        Albumentations compose object with validation transforms.
    """
    transforms = [
        # Resize to fixed size (required for batching)
        A.Resize(height=116, width=146),
        # Only normalization for validation
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ]
    return A.Compose(transforms)
