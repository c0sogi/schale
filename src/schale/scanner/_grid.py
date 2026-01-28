"""Grid and cell detection for inventory screenshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from schale.scanner._preprocessing import to_grayscale

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CellRegion:
    """A detected grid cell in the screenshot."""

    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    def extract_roi(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Extract the cell's region of interest from the source image."""
        return image[self.y : self.y + self.height, self.x : self.x + self.width]  # type: ignore[return-value]


def _cluster_rows(
    rects: list[tuple[int, int, int, int]],
    median_height: float,
) -> list[list[tuple[int, int, int, int]]]:
    """Group bounding rects into rows based on y-center proximity.

    Args:
        rects: List of (x, y, w, h) tuples sorted by y.
        median_height: Median cell height for proximity threshold.

    Returns:
        List of rows, each row being a list of (x, y, w, h) sorted by x.
    """
    if not rects:
        return []

    threshold = median_height * 0.3
    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = [rects[0]]
    current_y_center = rects[0][1] + rects[0][3] / 2.0

    for rect in rects[1:]:
        y_center = rect[1] + rect[3] / 2.0
        if abs(y_center - current_y_center) <= threshold:
            current_row.append(rect)
            # Update running average y center
            current_y_center = sum(r[1] + r[3] / 2.0 for r in current_row) / len(
                current_row
            )
        else:
            # Sort current row by x and start new row
            current_row.sort(key=lambda r: r[0])
            rows.append(current_row)
            current_row = [rect]
            current_y_center = y_center

    current_row.sort(key=lambda r: r[0])
    rows.append(current_row)
    return rows


def detect_grid(image: NDArray[np.uint8]) -> list[CellRegion]:
    """Detect the grid of item cells in an inventory screenshot.

    Uses adaptive thresholding and contour analysis to find
    card-shaped cells in the screenshot, then sorts them into
    a row/column grid.

    Args:
        image: BGR screenshot image.

    Returns:
        List of CellRegion objects sorted by (row, col).

    Raises:
        RuntimeError: If no valid grid cells are detected.
    """
    gray = to_grayscale(image)
    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    # Scale blockSize with image width (range: 11-51, must be odd)
    block_size = max(11, min(51, int(img_w / 40)))
    if block_size % 2 == 0:  # Must be odd for adaptiveThreshold
        block_size += 1

    logger.debug(
        "Image size: %dx%d, scaled parameters: blockSize=%d, kernelSize will be calculated",
        img_w,
        img_h,
        block_size,
    )

    # Adaptive threshold to separate bright card regions from background
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=-10,
    )

    # Scale kernel with image width (range: 3-5)
    kernel_size = max(3, min(5, int(img_w / 200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Morphological operations to clean up
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale min_area threshold based on image size
    # Use 0.1% for small images (<1200px), 0.5% for larger images
    # This allows for smaller cells in lower-resolution screenshots
    min_area_ratio = 0.001 if img_w < 1200 else 0.005
    min_area = img_area * min_area_ratio
    max_area = img_area * 0.10  # 10% of image

    candidate_rects: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or area > max_area:
            continue
        aspect_ratio = w / h if h > 0 else 0.0
        # Allow wider aspect ratio range (0.60-1.50) to accommodate various cell shapes
        # and different rendering resolutions
        if not (0.60 <= aspect_ratio <= 1.50):
            continue
        candidate_rects.append((x, y, w, h))

    if not candidate_rects:
        raise RuntimeError("No grid cells detected in the image")

    # Cluster by area: keep rects within +/- 40% of median area
    areas = sorted(r[2] * r[3] for r in candidate_rects)
    median_area = areas[len(areas) // 2]
    lower_area = median_area * 0.6
    upper_area = median_area * 1.4

    filtered_rects = [
        r for r in candidate_rects if lower_area <= r[2] * r[3] <= upper_area
    ]

    if not filtered_rects:
        # Fall back to all candidates if area filtering is too aggressive
        filtered_rects = candidate_rects
        logger.warning("Area clustering removed all cells; using unfiltered candidates")

    # Sort by y coordinate for row clustering
    filtered_rects.sort(key=lambda r: r[1])

    # Compute median height for row clustering threshold
    heights = sorted(r[3] for r in filtered_rects)
    median_height = float(heights[len(heights) // 2])

    # Cluster into rows
    rows = _cluster_rows(filtered_rects, median_height)

    # Build CellRegion list
    cells: list[CellRegion] = []
    for row_idx, row in enumerate(rows):
        for col_idx, (x, y, w, h) in enumerate(row):
            cells.append(
                CellRegion(
                    row=row_idx,
                    col=col_idx,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            )

    logger.debug(
        "Detected %d cells in %d rows (grid: %dx%d)",
        len(cells),
        len(rows),
        len(rows),
        max((len(r) for r in rows), default=0),
    )
    return cells
