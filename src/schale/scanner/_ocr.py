"""OCR utilities for reading tier badges and quantity text from cells."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, ClassVar, cast

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TIER_PATTERN = re.compile(r"[Tt](\d{1,2})")


class _OCRReaderHolder:
    """Lazy singleton for the EasyOCR reader to avoid slow re-initialization."""

    _reader: ClassVar[object | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(cls) -> object:
        """Return the shared EasyOCR Reader instance, creating it on first call."""
        if cls._reader is None:
            with cls._lock:
                if cls._reader is None:
                    import easyocr  # type: ignore[import-untyped]

                    cls._reader = easyocr.Reader(["en"], gpu=False)
        return cls._reader


def _run_ocr(
    image: NDArray[np.uint8] | cv2.typing.MatLike,
    allowlist: str,
) -> list[str]:
    """Run EasyOCR on a preprocessed image and return recognized texts.

    Args:
        image: Grayscale or BGR image to OCR.
        allowlist: Characters to restrict recognition to.

    Returns:
        List of recognized text strings.
    """
    reader = _OCRReaderHolder.get()
    # easyocr.Reader.readtext returns list of (bbox, text, confidence)
    raw_results = cast(
        list[list[Any]],
        reader.readtext(  # type: ignore[union-attr]
            image,
            allowlist=allowlist,
            paragraph=False,
        ),
    )
    return [str(entry[1]) for entry in raw_results]


def detect_exp_sphere_tier(cell_roi: NDArray[np.uint8]) -> int | None:
    """Detect if this is an Exp sphere based on dominant sphere color.

    Exp spheres don't have tier badges, but can be identified by their
    dominant color. In the schaledb equipment database, all Exp items
    use tier=0, regardless of their visual color (yellow/blue/purple).
    The specific Exp type is encoded in the icon name suffix.

    This function returns tier=0 if an Exp sphere is detected, which
    allows template matching to proceed against the Exp item database.

    Args:
        cell_roi: BGR cell region of interest.

    Returns:
        0 if an Exp sphere is detected, None otherwise.
    """
    h, w = cell_roi.shape[:2]

    # Sample the central icon area where the sphere is most prominent
    icon_roi = cell_roi[
        int(h * 0.25) : int(h * 0.75),
        int(w * 0.25) : int(w * 0.75),
    ]
    if icon_roi.size == 0:
        logger.debug("Color detection: icon_roi is empty")
        return None

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2HSV)

    # Require sufficient saturation and value to avoid white/gray/black regions
    lower_sat_val = np.array([0, 60, 60], dtype=np.uint8)
    upper_sat_val = np.array([180, 255, 255], dtype=np.uint8)
    sat_mask: NDArray[np.uint8] = cv2.inRange(hsv, lower_sat_val, upper_sat_val)  # type: ignore[assignment]

    # Require significant saturated pixels for Exp sphere detection
    # Exp spheres have very uniform, saturated colors (40% minimum)
    # Regular equipment typically has less saturated, more varied colors
    saturated_pixels = cv2.countNonZero(sat_mask)
    sat_percent = 100 * saturated_pixels / sat_mask.size
    if saturated_pixels < 0.40 * sat_mask.size:
        logger.debug("Color detection: insufficient saturated pixels (%d, %.1f%%)", saturated_pixels, sat_percent)
        return None

    # Analyze hue histogram for dominant color
    hue_channel = hsv[:, :, 0]
    masked_hues = hue_channel[sat_mask > 0]
    if masked_hues.size == 0:
        logger.debug("Color detection: no hues after masking")
        return None

    # Count pixels in each tier color range
    purple_mask = (masked_hues >= 135) & (masked_hues <= 160)
    blue_mask = (masked_hues >= 90) & (masked_hues < 135)
    yellow_orange_mask = (masked_hues >= 10) & (masked_hues <= 40)

    purple_count = np.sum(purple_mask)
    blue_count = np.sum(blue_mask)
    yellow_orange_count = np.sum(yellow_orange_mask)

    # Determine tier based on dominant color
    max_count = max(purple_count, blue_count, yellow_orange_count)
    total_colored = purple_count + blue_count + yellow_orange_count

    logger.debug(
        "Color detection: purple=%d (%.1f%%), blue=%d (%.1f%%), yellow=%d (%.1f%%), sat_px=%d",
        purple_count, 100 * purple_count / saturated_pixels if saturated_pixels > 0 else 0,
        blue_count, 100 * blue_count / saturated_pixels if saturated_pixels > 0 else 0,
        yellow_orange_count, 100 * yellow_orange_count / saturated_pixels if saturated_pixels > 0 else 0,
        saturated_pixels,
    )

    # Require strong dominance of a single tier color
    # Exp spheres have very pure, uniform colors (30% of saturated pixels minimum)
    if max_count < 0.30 * saturated_pixels:
        logger.debug("Color detection: dominant color below 30%% threshold")
        return None

    # Require the dominant color to be at least 70% of all tier colors
    # This prevents false positives on multi-colored equipment
    if total_colored > 0 and max_count < 0.70 * total_colored:
        logger.debug("Color detection: dominant color below 70%% of tier colors")
        return None

    # All Exp items in schaledb use tier=0
    # The specific type (yellow/blue/purple) is encoded in icon name
    if max_count == purple_count:
        logger.debug("Color detection: detected Exp sphere (purple) -> tier 0")
        return 0
    elif max_count == blue_count:
        logger.debug("Color detection: detected Exp sphere (blue) -> tier 0")
        return 0
    elif max_count == yellow_orange_count:
        logger.debug("Color detection: detected Exp sphere (yellow/orange) -> tier 0")
        return 0

    return None


def read_tier_badge(cell_roi: NDArray[np.uint8]) -> int | None:
    """Read the tier badge (T2-T10) from the lower-left corner of a cell.

    The tier badge is white text on a blue rounded rectangle
    located in the bottom-left area of each equipment card.

    If no tier badge is found, falls back to color-based detection
    for Exp spheres which don't have text badges.

    Args:
        cell_roi: BGR cell region of interest.

    Returns:
        Tier number (1-10) or None if no valid badge is found.
    """
    h, w = cell_roi.shape[:2]

    # Crop the lower-left region where the tier badge appears
    badge_roi = cell_roi[int(h * 0.75) : h, 0 : int(w * 0.40)]
    if badge_roi.size == 0:
        return None

    # Detect blue badge background via HSV
    hsv = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 80, 80], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    blue_mask: NDArray[np.uint8] = cv2.inRange(hsv, lower_blue, upper_blue)  # type: ignore[assignment]

    # If not enough blue pixels, no badge present
    if cv2.countNonZero(blue_mask) < 0.15 * blue_mask.size:
        # No badge detected - will try Exp sphere detection at end of function
        logger.debug("No tier badge background detected")
        # Don't return here - let OCR attempts proceed first
        tier_from_ocr = None
    else:

        # Extract the blue badge region from the original BGR image
        badge_region_bgr: NDArray[np.uint8] = cv2.bitwise_and(badge_roi, badge_roi, mask=blue_mask)  # type: ignore[assignment]

        # Convert to grayscale and threshold for white text (bright pixels)
        gray_badge = cv2.cvtColor(badge_region_bgr, cv2.COLOR_BGR2GRAY)
        _, text_only = cv2.threshold(gray_badge, 200, 255, cv2.THRESH_BINARY)

        # Dilate to connect broken text before upscaling
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        text_only = cv2.dilate(text_only, dilate_kernel, iterations=1)  # type: ignore[assignment]

        # Upscale for better OCR accuracy
        # Upscale more aggressively - aim for at least 120px width
        scale = max(2, 120 // max(text_only.shape[1], 1))
        if scale > 1:
            text_only = cv2.resize(  # type: ignore[assignment]
                text_only,
                (text_only.shape[1] * scale, text_only.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )

        texts = _run_ocr(text_only, allowlist="T0123456789")
        for text in texts:
            match = _TIER_PATTERN.search(text)
            if match:
                tier = int(match.group(1))
                if 1 <= tier <= 10:
                    return tier

        # Handle common OCR misreads
        raw = "".join(texts).upper().strip()
        raw = raw.replace("Z", "2").replace("O", "0").replace("I", "1").replace("S", "5")
        match = _TIER_PATTERN.search(raw)
        if match:
            tier = int(match.group(1))
            if 1 <= tier <= 10:
                return tier

        logger.debug("Could not parse tier badge from OCR results: %s", texts)
        tier_from_ocr = None

    # If badge detection and OCR both failed, try Exp sphere color detection as fallback
    # This catches Exp spheres which have no tier badge at all
    if tier_from_ocr is None:
        return None  # Temporarily disabled for testing
        # return detect_exp_sphere_tier(cell_roi)

    return tier_from_ocr


def _parse_quantity_text(raw: str) -> int | None:
    """Parse a quantity string like 'x104', 'x13K', '558' into an integer.

    Args:
        raw: Raw OCR text for the quantity.

    Returns:
        Integer quantity or None if unparseable.
    """
    text = raw.strip().lower().replace(",", "").replace(" ", "")
    if text.startswith("x"):
        text = text[1:]
    if not text:
        return None

    if text.endswith("k"):
        try:
            return int(float(text[:-1]) * 1000)
        except ValueError:
            return None

    # Remove any trailing non-digit characters
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def read_quantity(cell_roi: NDArray[np.uint8]) -> int | None:
    """Read the quantity text (x###) from the bottom area of a cell.

    The quantity is displayed as white outlined text near the bottom
    of each card, typically right of the tier badge.

    Args:
        cell_roi: BGR cell region of interest.

    Returns:
        Integer quantity or None if unreadable.
    """
    h, w = cell_roi.shape[:2]

    # Crop the bottom region where quantity text appears
    qty_roi = cell_roi[int(h * 0.78) : h, int(w * 0.25) :]
    if qty_roi.size == 0:
        return None

    # Convert to grayscale and threshold for white text
    gray_qty = cv2.cvtColor(qty_roi, cv2.COLOR_BGR2GRAY)

    # Upscale small regions for better OCR
    min_width = 80
    if gray_qty.shape[1] < min_width:
        scale = min_width // max(gray_qty.shape[1], 1) + 1
        gray_qty = cv2.resize(  # type: ignore[assignment]
            gray_qty,
            (gray_qty.shape[1] * scale, gray_qty.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    _, text_mask = cv2.threshold(gray_qty, 180, 255, cv2.THRESH_BINARY)

    # Dilate slightly to connect broken strokes
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.dilate(text_mask, dilate_kernel, iterations=1)

    texts = _run_ocr(text_mask, allowlist="x0123456789Kk.,")

    for text in texts:
        result = _parse_quantity_text(text)
        if result is not None:
            return result

    # Try joining all texts as a single string
    joined = "".join(texts)
    result = _parse_quantity_text(joined)
    if result is not None:
        return result

    logger.debug("Could not parse quantity from OCR results: %s", texts)
    return None
