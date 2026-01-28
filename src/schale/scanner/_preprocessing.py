"""Image preprocessing utilities for the inventory scanner."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from numpy.typing import NDArray

ImageSource = Union[str, Path, NDArray[np.uint8]]


def load_image(source: ImageSource) -> NDArray[np.uint8]:
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
        return source  # type: ignore[return-value]
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img_buffer = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img  # type: ignore[return-value]


def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR image to grayscale, or return as-is if already single-channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore[return-value]


def to_hsv(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # type: ignore[return-value]


def resize_to_width(image: NDArray[np.uint8], target_width: int) -> NDArray[np.uint8]:
    """Resize image to a target width, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)  # type: ignore[return-value]


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
