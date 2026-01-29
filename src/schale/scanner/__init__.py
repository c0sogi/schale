"""Inventory scanner module for Blue Archive screenshots.

Recognizes equipment items and their quantities from in-game
inventory screenshots using OpenCV template matching and EasyOCR.

Requires optional dependencies::

    uv add schale[scanner]

Usage::

    from schale.scanner import scan_inventory

    result = scan_inventory("screenshot.png")
    for item in result.items:
        print(f"{item.icon_name} T{item.tier} x{item.quantity}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, cast

try:
    import cv2
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise ImportError(
        "The scanner module requires additional dependencies. "
        "Install them with: uv add schale[scanner]"
    ) from exc

from schale import literal
from schale.scanner._grid import CellRegion, detect_grid
from schale.scanner._icons import IconAtlas
from schale.scanner._ocr import read_quantity, read_tier_badge
from schale.scanner._preprocessing import (
    compute_blue_ratio,
    detect_blueprint_background,
    load_image,
    preprocess_blueprint_cell,
)
from schale.schema.scanner import ScannedItem, ScanResult

logger = logging.getLogger(__name__)

ImageSource = Union[str, Path, "NDArray[np.uint8]"]

# Singleton icon atlas, prepared lazily on first scan
_atlas: IconAtlas | None = None


def _get_atlas(*, force_download: bool = False) -> IconAtlas:
    """Return the global IconAtlas, preparing it on first call."""
    global _atlas  # noqa: PLW0603
    if _atlas is None or force_download:
        _atlas = IconAtlas()
        _atlas.prepare(force_download=force_download)
    return _atlas


def _process_cell(
    cell: CellRegion,
    image: NDArray[np.uint8],
    atlas: IconAtlas,
    confidence_threshold: float,
) -> ScannedItem | None:
    """Process a single grid cell to identify the equipment item.

    Args:
        cell: The detected cell region.
        image: The full BGR source image.
        atlas: Prepared icon atlas for template matching.
        confidence_threshold: Minimum confidence for a valid match.

    Returns:
        ScannedItem if successfully identified, None otherwise.
    """
    cell_roi = cell.extract_roi(image)

    # Ensure exact template size for matching (normalization may have small variance)
    TARGET_H, TARGET_W = 116, 146
    if cell_roi.shape[:2] != (TARGET_H, TARGET_W):
        cell_roi = cv2.resize(
            cell_roi,
            (TARGET_W, TARGET_H),  # Note: cv2.resize takes (width, height)
            interpolation=cv2.INTER_AREA,
        )

    # Classify cell type (blueprint vs regular)
    is_blueprint = detect_blueprint_background(cell_roi)

    # Apply gentle CLAHE preprocessing for blueprint cells
    # (using simplified approach - 4-stage pipeline was too aggressive)
    if is_blueprint:
        blue_ratio = compute_blue_ratio(cell_roi)
        if blue_ratio > 0.24:
            logger.debug(
                "Cell (%d, %d) has high blue ratio (%.3f), applying gentle CLAHE preprocessing",
                cell.row,
                cell.col,
                blue_ratio,
            )
            cell_roi = preprocess_blueprint_cell(cell_roi)

    # Read tier badge (pass is_blueprint to avoid false Exp detection on blueprints)
    tier = read_tier_badge(cell_roi, is_blueprint=is_blueprint)

    # Read quantity
    quantity = read_quantity(cell_roi)
    if quantity is None:
        quantity = 1
        logger.debug(
            "Could not read quantity for cell (%d, %d), defaulting to 1",
            cell.row,
            cell.col,
        )

    # Use CNN classifier to predict icon (category + tier)
    # CNN is trained on clean icons, so it works better than template/feature matching
    # which struggle with tier badges and quantity text overlays
    if atlas._cnn is not None:
        predictions = atlas._cnn.predict(cell_roi, top_k=1)
        if predictions:
            pred_name, cnn_confidence = predictions[0]

            # Look up the predicted template directly
            best_match = atlas._templates.get(pred_name)

            if best_match is not None:
                logger.info(
                    "CNN match: %s (conf=%.3f)",
                    pred_name,
                    cnn_confidence,
                )
            else:
                logger.warning(
                    "CNN predicted %s but not found in templates",
                    pred_name,
                )
                return None
        else:
            logger.warning("CNN returned no predictions")
            return None
    else:
        logger.error("CNN classifier not enabled")
        return None

    return ScannedItem(
        equipment_id=best_match.equipment_id,
        category=cast("literal.EquipmentCategory", best_match.category),
        tier=best_match.tier,
        quantity=quantity,
        is_blueprint=best_match.is_blueprint,
        icon_name=best_match.icon_name,
        confidence=round(cnn_confidence, 4),
        grid_position=(cell.row, cell.col),
    )


def scan_inventory(
    image: ImageSource,
    *,
    confidence_threshold: float = 0.15,
    force_icon_download: bool = False,
) -> ScanResult:
    """Scan a Blue Archive inventory screenshot and identify equipment items.

    Uses multi-tier matching strategy:
    1. Template matching with alpha masks (primary, background-invariant)
    2. KAZE feature matching (final fallback)

    Args:
        image: Path to screenshot file, or numpy array (BGR format).
        confidence_threshold: Minimum template matching confidence (0.0-1.0).
            Items below this threshold are reported as unrecognized.
        force_icon_download: Re-download all reference icons even if cached.

    Returns:
        ScanResult with identified items and grid metadata.

    Raises:
        FileNotFoundError: If image path does not exist.
        ValueError: If image cannot be decoded.
        RuntimeError: If grid detection fails entirely.
        ImportError: If CNN model or dependencies are not available.
    """
    # Load image
    img = load_image(image)
    original_h, original_w = img.shape[:2]

    # First pass: detect grid at original resolution
    logger.debug("First pass grid detection at %dx%d", original_w, original_h)
    cells_pass1 = detect_grid(img)

    if not cells_pass1:
        raise RuntimeError("No grid cells detected in the image")

    # Calculate normalization scale based on detected cell size
    # Native template size is 116h x 146w (typical SchaleDB icon dimensions)
    TEMPLATE_WIDTH = 146
    TEMPLATE_HEIGHT = 116

    # Get median cell dimensions
    widths = [c.width for c in cells_pass1]
    heights = [c.height for c in cells_pass1]
    median_width = sorted(widths)[len(widths) // 2]
    median_height = sorted(heights)[len(heights) // 2]

    # Calculate separate scale factors for width and height
    # This allows anisotropic scaling to match template aspect ratio exactly
    scale_w = TEMPLATE_WIDTH / median_width
    scale_h = TEMPLATE_HEIGHT / median_height

    logger.info(
        "Cell size: %dx%d, template size: %dx%d, scale: w=%.3f h=%.3f",
        median_width,
        median_height,
        TEMPLATE_WIDTH,
        TEMPLATE_HEIGHT,
        scale_w,
        scale_h,
    )

    # Rescale image if scale factors are significantly different from 1.0
    # Use anisotropic scaling to match template aspect ratio exactly
    needs_scaling = abs(scale_w - 1.0) > 0.1 or abs(scale_h - 1.0) > 0.1

    if needs_scaling:
        new_width = int(original_w * scale_w)
        new_height = int(original_h * scale_h)
        img = cast(
            NDArray[np.uint8],
            cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC),
        )
        logger.info(
            "Normalized image from %dx%d to %dx%d (anisotropic scale: w=%.3f h=%.3f)",
            original_w,
            original_h,
            new_width,
            new_height,
            scale_w,
            scale_h,
        )

        # Scale the cell coordinates with separate factors
        # This preserves all detected cells from the first pass
        cells = [
            CellRegion(
                row=c.row,
                col=c.col,
                x=int(c.x * scale_w),
                y=int(c.y * scale_h),
                width=int(c.width * scale_w),
                height=int(c.height * scale_h),
            )
            for c in cells_pass1
        ]
        logger.debug("Scaled %d cell coordinates with anisotropic scaling", len(cells))
    else:
        logger.info("Image already at optimal scale, skipping normalization")
        cells = cells_pass1

    # Prepare icon atlas and enable CNN classifier
    atlas = _get_atlas(force_download=force_icon_download)
    atlas.enable_cnn()
    logger.info("CNN classifier enabled")

    # Ensure we have cells after potential re-detection
    if not cells:
        raise RuntimeError("No grid cells detected in the image")

    # Process each cell
    items: list[ScannedItem] = []
    unrecognized: list[tuple[int, int]] = []

    for cell in cells:
        result = _process_cell(cell, img, atlas, confidence_threshold)
        if result is not None:
            items.append(result)
        else:
            unrecognized.append((cell.row, cell.col))

    # Compute grid dimensions
    max_row = max((c.row for c in cells), default=0)
    max_col = max((c.col for c in cells), default=0)

    logger.info(
        "Scan complete: %d items identified, %d unrecognized, grid %dx%d",
        len(items),
        len(unrecognized),
        max_row + 1,
        max_col + 1,
    )

    return ScanResult(
        items=items,
        unrecognized_cells=unrecognized,
        grid_dimensions=(max_row + 1, max_col + 1),
        source_resolution=(original_h, original_w),
    )


__all__ = [
    "ScanResult",
    "ScannedItem",
    "scan_inventory",
]
