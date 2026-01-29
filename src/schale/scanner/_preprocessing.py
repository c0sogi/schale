"""Image preprocessing utilities for the inventory scanner."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from numpy.typing import NDArray

ImageSource = Union[str, Path, NDArray[np.uint8]]


def load_image(source: ImageSource) -> np.ndarray:
    """Load an image from a file path or pass through a numpy array.

    Args:
        source: Path to an image file, or a BGR numpy array.

    Returns:
        BGR uint8 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if isinstance(source, np.ndarray):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img_buffer = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def to_grayscale(image: NDArray[np.uint8]) -> np.ndarray:
    """Convert BGR image to grayscale, or return as-is if already single-channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: NDArray[np.uint8]) -> np.ndarray:
    """Convert BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def resize_to_width(image: NDArray[np.uint8], target_width: int) -> np.ndarray:
    """Resize image to a target width, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def detect_blueprint_background(cell_roi: NDArray[np.uint8]) -> bool:
    """Detect if a cell has the blue blueprint background.

    Blueprint items in Blue Archive have a distinctive blue-tinted
    grid-paper background compared to regular items.

    Args:
        cell_roi: BGR cell region of interest.

    Returns:
        True if the cell has a blue blueprint background.
    """
    hsv = to_hsv(cell_roi)
    h, w = cell_roi.shape[:2]
    # Sample the center region to avoid edges and overlay text
    center = hsv[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
    lower_blue = np.array([95, 50, 50], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(center, lower_blue, upper_blue)
    blue_ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)
    return blue_ratio > 0.25


def compute_blue_ratio(cell_roi: NDArray[np.uint8]) -> float:
    """Compute the ratio of blue pixels in a cell region.

    Used to detect cells with high blue background interference (blueprint grid overlay).

    Args:
        cell_roi: BGR cell region of interest.

    Returns:
        Float in range [0.0, 1.0] representing the proportion of blue pixels.
    """
    hsv = to_hsv(cell_roi)
    h, w = cell_roi.shape[:2]
    # Sample the center region to avoid edges and overlay text
    center = hsv[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
    lower_blue = np.array([95, 50, 50], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(center, lower_blue, upper_blue)
    blue_ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)
    return blue_ratio


def preprocess_blueprint_cell(cell_roi: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply gentle CLAHE normalization for blueprint cells.

    Previous 4-stage pipeline (blue suppression, CLAHE, morphological grid removal, denoising)
    was too aggressive and destroyed icon features, causing recognition to drop from 72% to 52%.

    This simplified approach only applies CLAHE normalization to enhance local contrast
    without suppressing color channels or removing morphological features.

    Args:
        cell_roi: BGR cell region of interest with blueprint background.

    Returns:
        Preprocessed BGR cell image with enhanced contrast.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.debug("Applying gentle CLAHE normalization for blueprint cell")

    # Convert to LAB color space
    lab = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to L channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    bgr_clahe_raw = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    bgr_clahe = np.asarray(bgr_clahe_raw, dtype=np.uint8)

    logger.debug("Gentle CLAHE preprocessing complete")
    return bgr_clahe


__all__ = [
    "load_image",
    "to_grayscale",
    "to_hsv",
    "resize_to_width",
    "detect_blueprint_background",
    "compute_blue_ratio",
    "preprocess_blueprint_cell",
    "ImageSource",
]
