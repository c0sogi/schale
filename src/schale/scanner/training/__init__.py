"""Training utilities for equipment icon classification model.

Provides:
- Class mappings from SchaleDB equipment data for model training and inference
- PyTorch Dataset for synthetic training data generation
- DataLoader creation utilities with train/val splits
- Augmentation pipelines for data augmentation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from schale.cache_control import cache_collection

logger = logging.getLogger(__name__)

# Same categories as IconAtlas for consistency
_TEMPLATE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Hat",
        "Gloves",
        "Shoes",
        "Bag",
        "Badge",
        "Hairpin",
        "Charm",
        "Watch",
        "Necklace",
        "Exp",
        "WeaponExpGrowthA",
        "WeaponExpGrowthB",
        "WeaponExpGrowthC",
        "WeaponExpGrowthZ",
    }
)


class ClassMapping(TypedDict):
    """Type for class mapping JSON structure."""

    class_to_name: dict[str, str]  # "0" -> "equipment_icon_hat_tier1"
    name_to_class: dict[str, int]  # "equipment_icon_hat_tier1" -> 0
    num_classes: int  # Total number of classes (191)
    categories: list[str]  # List of included categories


def generate_class_mapping(output_path: Path | None = None) -> ClassMapping:
    """Generate class mapping from SchaleDB equipment data.

    Creates a bidirectional mapping between class indices (0-190) and equipment
    icon names for the 191 equipment types in _TEMPLATE_CATEGORIES.

    The mapping is deterministic (sorted by icon_name) to ensure consistency
    across training and inference.

    Args:
        output_path: Optional path to save the mapping JSON file.
            Defaults to src/schale/scanner/models/class_mapping.json

    Returns:
        ClassMapping dict with class_to_name, name_to_class, num_classes, and categories.

    Example:
        >>> mapping = generate_class_mapping()
        >>> mapping["num_classes"]
        191
        >>> mapping["class_to_name"]["0"]
        'equipment_icon_badge_tier1'
        >>> mapping["name_to_class"]["equipment_icon_badge_tier1"]
        0
    """
    # Read equipment data from SchaleDB cache
    equipments = cache_collection.equipments

    # Extract all equipment items with Category in _TEMPLATE_CATEGORIES
    equipment_icons: list[str] = []

    for eq_id, eq in equipments.items():
        if eq.Category not in _TEMPLATE_CATEGORIES:
            continue

        icon_name = eq.Icon
        equipment_icons.append(icon_name)

    # Sort icon names for deterministic ordering
    # This ensures class 0 is always the same equipment across runs
    equipment_icons_sorted = sorted(set(equipment_icons))

    # Create bidirectional mappings
    class_to_name = {str(i): name for i, name in enumerate(equipment_icons_sorted)}
    name_to_class = {name: i for i, name in enumerate(equipment_icons_sorted)}

    num_classes = len(equipment_icons_sorted)

    mapping: ClassMapping = {
        "class_to_name": class_to_name,
        "name_to_class": name_to_class,
        "num_classes": num_classes,
        "categories": sorted(_TEMPLATE_CATEGORIES),
    }

    # Save to file if path provided
    if output_path is None:
        output_path = Path(__file__).parent.parent / "models" / "class_mapping.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logger.info(
        "Generated class mapping: %d classes saved to %s",
        num_classes,
        output_path,
    )

    return mapping


def load_class_mapping(path: Path | None = None) -> ClassMapping:
    """Load class mapping from JSON file.

    Args:
        path: Optional path to the mapping JSON file.
            Defaults to src/schale/scanner/models/class_mapping.json

    Returns:
        ClassMapping dict with class_to_name, name_to_class, num_classes, and categories.

    Raises:
        FileNotFoundError: If mapping file does not exist.
        ValueError: If mapping file is invalid.
    """
    if path is None:
        path = Path(__file__).parent.parent / "models" / "class_mapping.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {path}\n"
            "Generate it first by calling generate_class_mapping()"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    required_keys = {"class_to_name", "name_to_class", "num_classes", "categories"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"Invalid class mapping file: missing keys {required_keys - data.keys()}"
        )

    # Convert name_to_class values to int (JSON stores all keys as strings)
    data["name_to_class"] = {k: int(v) for k, v in data["name_to_class"].items()}

    return data


# Import after function definitions to avoid circular imports
from schale.scanner.training.dataset import (  # noqa: E402
    EquipmentDataset,
    create_dataloaders,
)
from schale.scanner.training.train import TrainConfig, train  # noqa: E402

__all__ = [
    "ClassMapping",
    "generate_class_mapping",
    "load_class_mapping",
    "EquipmentDataset",
    "create_dataloaders",
    "TrainConfig",
    "train",
]
