"""PyTorch Dataset for synthetic training data from SchaleDB icons."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, cast

import albumentations as A
import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, random_split

from schale.scanner.training import load_class_mapping
from schale.scanner.training.augment import get_train_transform, get_val_transform

logger = logging.getLogger(__name__)

_ICON_CACHE_DIR = (
    Path(os.environ.get("SCHALE_CACHE_DIR") or (Path.home() / ".schale" / "cache"))
    / "icons"
)


def _load_icon_rgb(path: Path) -> NDArray[np.uint8] | None:
    """Load icon as RGB image with unicode path support.

    Loads RGBA WebP icons, composites over gray background (240, 240, 240),
    converts to RGB format (H, W, C) for albumentations.

    Args:
        path: Path to the icon file (WebP or PNG).

    Returns:
        RGB image as numpy array (H, W, 3) with uint8 dtype, or None if loading fails.
    """
    # Use np.fromfile for unicode path support
    img_buffer = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("Cannot decode icon: %s", path)
        return None

    # Handle alpha channel if present (BGRA -> RGB)
    if img.shape[-1] == 4:  # BGRA
        bgr = img[..., :3]
        alpha_f = img[..., 3:4] / 255.0
        background = np.full_like(bgr, 240, dtype=np.uint8)
        composited = (bgr * alpha_f + background * (1 - alpha_f)).astype(np.uint8)
        rgb = cast(NDArray[np.uint8], cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
    elif len(img.shape) == 3:  # BGR, no alpha
        rgb = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:  # Grayscale -> convert to RGB
        rgb = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    return rgb


class EquipmentDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset for equipment icon classification.

    Generates synthetic training data by loading SchaleDB reference icons
    and applying augmentation transforms. Each icon generates multiple
    augmented samples per epoch.

    Args:
        augment_factor: Number of augmented samples to generate per icon.
        transform: Albumentations transform pipeline. Defaults to training transform.
        split: Dataset split - "train" or "val". Determines default transform.
        seed: Random seed for reproducibility.
        target_size: Resize icons to (height, width). If None, keep native resolution.

    Attributes:
        num_classes: Total number of equipment classes (191).
        class_to_name: Mapping from class index to icon name.
        name_to_class: Mapping from icon name to class index.
    """

    def __init__(
        self,
        augment_factor: int = 100,
        transform: A.Compose | None = None,
        split: Literal["train", "val"] = "train",
        seed: int = 42,
        target_size: tuple[int, int] | None = (116, 146),  # (H, W)
    ) -> None:
        """Initialize EquipmentDataset."""
        super().__init__()

        self.augment_factor = augment_factor
        self.split = split
        self.seed = seed
        self.target_size = target_size

        # Load class mapping
        mapping = load_class_mapping()
        self.num_classes: int = mapping["num_classes"]
        self.class_to_name: dict[str, str] = mapping["class_to_name"]
        self.name_to_class: dict[str, int] = mapping["name_to_class"]

        # Set transform
        if transform is None:
            if split == "train":
                self.transform: A.Compose = get_train_transform()
            else:
                self.transform = get_val_transform()
        else:
            self.transform = transform

        # Load all icons from cache
        self.icons: list[tuple[NDArray[np.uint8], int]] = []
        self._load_icons()

        # Set random seed for reproducibility
        np.random.seed(self.seed)

    def _load_icons(self) -> None:
        """Load all equipment icons from cache directory.

        Populates self.icons with (image, class_label) pairs.
        """
        if not _ICON_CACHE_DIR.exists():
            raise FileNotFoundError(
                f"Icon cache directory not found: {_ICON_CACHE_DIR}\n"
                "Download icons first by calling IconAtlas.prepare()"
            )

        loaded = 0
        failed = 0

        for icon_name, class_idx in self.name_to_class.items():
            # Try .webp first, then .png as fallback
            icon_path = None
            for ext in (".webp", ".png"):
                candidate_path = _ICON_CACHE_DIR / f"{icon_name}{ext}"
                if candidate_path.exists():
                    icon_path = candidate_path
                    break

            if icon_path is None:
                logger.warning("Icon not found: %s", icon_name)
                failed += 1
                continue

            # Load icon as RGB
            rgb = _load_icon_rgb(icon_path)
            if rgb is None:
                failed += 1
                continue

            # Resize to target size if specified
            if self.target_size is not None:
                h, w = self.target_size
                rgb = cast(
                    NDArray[np.uint8],
                    cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA),
                )

            self.icons.append((rgb, class_idx))
            loaded += 1

        if loaded == 0:
            raise ValueError("No icons loaded! Check icon cache directory.")

        logger.info(
            "Loaded %d icons for %s dataset (%d failed)",
            loaded,
            self.split,
            failed,
        )

    def __len__(self) -> int:
        """Return total number of samples (num_icons * augment_factor)."""
        return len(self.icons) * self.augment_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get augmented image tensor and class label.

        Args:
            idx: Sample index (0 to len(self) - 1).

        Returns:
            (image_tensor, class_label) where image_tensor is (C, H, W) float tensor.
        """
        # Map idx to (icon_idx, augment_idx)
        icon_idx = idx // self.augment_factor
        augment_idx = idx % self.augment_factor

        # Get base icon and label
        image, label = self.icons[icon_idx]

        # Apply augmentation transform
        # Set deterministic seed per sample for reproducibility
        # but allow variation across augmentations of same icon
        np.random.seed(self.seed + icon_idx * self.augment_factor + augment_idx)

        # Albumentations expects RGB numpy array (H, W, C)
        augmented = self.transform(image=image)
        image_tensor: torch.Tensor = augmented["image"]  # (C, H, W) float tensor

        return image_tensor, label


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    augment_factor: int = 100,
    train_split: float = 0.8,
    seed: int = 42,
    target_size: tuple[int, int] | None = (116, 146),
) -> tuple[DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]]]:
    """Create train and validation DataLoaders with 80/20 split.

    Args:
        batch_size: Batch size for both train and val loaders.
        num_workers: Number of worker processes for data loading.
        augment_factor: Number of augmented samples per icon per epoch.
        train_split: Fraction of data to use for training (default 0.8).
        seed: Random seed for reproducibility.
        target_size: Resize icons to (height, width). If None, keep native resolution.

    Returns:
        (train_loader, val_loader) tuple.

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     batch_size=32,
        ...     num_workers=4,
        ...     augment_factor=100
        ... )
        >>> len(train_loader.dataset)
        15280  # 191 icons * 100 augment_factor * 0.8 train_split
        >>> len(val_loader.dataset)
        3820   # 191 icons * 100 augment_factor * 0.2 val_split
    """
    # Create full dataset with training transforms
    full_dataset = EquipmentDataset(
        augment_factor=augment_factor,
        split="train",
        seed=seed,
        target_size=target_size,
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Apply validation transform to val_dataset
    # Note: val_dataset uses the same base dataset, but we create a separate
    # EquipmentDataset with val transforms for proper behavior
    val_dataset_standalone = EquipmentDataset(
        augment_factor=augment_factor,
        split="val",
        seed=seed,
        target_size=target_size,
    )

    # Split val_dataset_standalone to match the same indices as val_dataset
    _, val_dataset_standalone = random_split(
        val_dataset_standalone,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create DataLoaders
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_dataset_standalone,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Created DataLoaders: train=%d samples, val=%d samples",
        len(train_dataset),
        len(val_dataset_standalone),
    )

    return train_loader, val_loader


__all__ = [
    "EquipmentDataset",
    "create_dataloaders",
]
