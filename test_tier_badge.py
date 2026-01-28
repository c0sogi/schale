"""Debug tier badge OCR."""

import cv2
import numpy as np
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid
from schale.scanner._ocr import read_tier_badge

img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

print(f"Testing tier badge reading on cells...\n")

# Test first 10 cells
for i in range(min(10, len(cells))):
    cell = cells[i]
    cell_roi = cell.extract_roi(img)

    h, w = cell_roi.shape[:2]
    badge_roi = cell_roi[int(h * 0.75) : h, 0 : int(w * 0.40)]

    # Check blue pixels
    hsv = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 80, 80], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(blue_mask) / max(blue_mask.size, 1)

    tier = read_tier_badge(cell_roi)

    print(f"Cell ({cell.row}, {cell.col}):")
    print(f"  Cell size: {cell_roi.shape}")
    print(f"  Badge ROI size: {badge_roi.shape}")
    print(f"  Blue pixel ratio: {blue_ratio:.3f} (threshold: 0.15)")
    print(f"  Tier result: {tier}")
    print()
