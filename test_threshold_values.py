"""Test different threshold values."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid

img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

cell = cells[4]  # (0, 4)
cell_roi = cell.extract_roi(img)

h, w = cell_roi.shape[:2]
badge_roi = cell_roi[int(h * 0.75) : h, 0 : int(w * 0.40)]

# Get blue mask
hsv = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2HSV)
lower_blue = np.array([95, 80, 80], dtype=np.uint8)
upper_blue = np.array([135, 255, 255], dtype=np.uint8)
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Convert to grayscale
gray_badge = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2GRAY)

print(f"Badge grayscale stats:")
print(f"  Min: {gray_badge.min()}")
print(f"  Max: {gray_badge.max()}")
print(f"  Mean: {gray_badge.mean():.1f}")
print(f"  Pixels > 200: {np.sum(gray_badge > 200)}")
print(f"  Pixels > 180: {np.sum(gray_badge > 180)}")
print(f"  Pixels > 160: {np.sum(gray_badge > 160)}")
print(f"  Pixels > 140: {np.sum(gray_badge > 140)}")
print()

# Test different thresholds
debug_dir = Path("B:/Projects/schale/debug_thresholds")
debug_dir.mkdir(exist_ok=True)

for thresh_val in [200, 180, 160, 140, 120]:
    _, text_mask = cv2.threshold(gray_badge, thresh_val, 255, cv2.THRESH_BINARY)
    text_only = cv2.bitwise_and(text_mask, blue_mask)

    nonzero = cv2.countNonZero(text_only)
    print(f"Threshold {thresh_val}: {nonzero} non-zero pixels in text_only")

    cv2.imwrite(str(debug_dir / f"text_mask_{thresh_val}.png"), text_mask)
    cv2.imwrite(str(debug_dir / f"text_only_{thresh_val}.png"), text_only)

print(f"\nDebug images saved to {debug_dir}")
