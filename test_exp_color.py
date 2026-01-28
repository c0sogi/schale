"""Test Exp sphere color-based tier detection."""
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from schale.scanner import scan_inventory
from schale.scanner._ocr import detect_exp_sphere_tier

# First, run the scan to get unrecognized cells
result = scan_inventory(
    r"C:\Users\cosogi\Downloads\제목 없음.png",
    confidence_threshold=0.03
)

print(f"\nTotal unrecognized cells: {len(result.unrecognized_cells)}")

# Test color detection on each unrecognized cell
for idx, cell in enumerate(result.unrecognized_cells):
    print(f"\n--- Unrecognized cell {idx} at row={cell.row}, col={cell.col} ---")

    # The cell image is stored in BGR format
    if cell.image is not None:
        tier = detect_exp_sphere_tier(cell.image)

        # Analyze the color distribution
        h, w = cell.image.shape[:2]
        icon_roi = cell.image[
            int(h * 0.25):int(h * 0.75),
            int(w * 0.25):int(w * 0.75),
        ]

        hsv = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2HSV)

        # Count saturated pixels
        lower_sat_val = np.array([0, 60, 60], dtype=np.uint8)
        upper_sat_val = np.array([180, 255, 255], dtype=np.uint8)
        sat_mask = cv2.inRange(hsv, lower_sat_val, upper_sat_val)
        saturated_pixels = cv2.countNonZero(sat_mask)

        hue_channel = hsv[:, :, 0]
        masked_hues = hue_channel[sat_mask > 0]

        if masked_hues.size > 0:
            purple_mask = (masked_hues >= 135) & (masked_hues <= 160)
            blue_mask = (masked_hues >= 90) & (masked_hues < 135)
            yellow_orange_mask = (masked_hues >= 10) & (masked_hues <= 40)

            purple_count = np.sum(purple_mask)
            blue_count = np.sum(blue_mask)
            yellow_orange_count = np.sum(yellow_orange_mask)

            print(f"  Saturated pixels: {saturated_pixels} ({100*saturated_pixels/sat_mask.size:.1f}%)")
            print(f"  Purple (T3): {purple_count} ({100*purple_count/saturated_pixels:.1f}%)")
            print(f"  Blue (T2): {blue_count} ({100*blue_count/saturated_pixels:.1f}%)")
            print(f"  Yellow/Orange (T1): {yellow_orange_count} ({100*yellow_orange_count/saturated_pixels:.1f}%)")
            print(f"  Detected tier: {tier}")

            # Save a debug image showing the icon ROI
            debug_path = f"debug_exp_cell_{idx}.png"
            cv2.imwrite(debug_path, icon_roi)
            print(f"  Saved debug image to {debug_path}")
        else:
            print(f"  No saturated pixels found")
            print(f"  Detected tier: {tier}")
