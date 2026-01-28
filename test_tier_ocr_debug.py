"""Debug tier badge OCR input."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid

img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

# Test cell (0, 4) which has 0.267 blue ratio
cell = cells[4]  # (0, 4)
cell_roi = cell.extract_roi(img)

h, w = cell_roi.shape[:2]
badge_roi = cell_roi[int(h * 0.75) : h, 0 : int(w * 0.40)]

print(f"Cell (0, 4) badge processing:")
print(f"  Badge ROI size: {badge_roi.shape}")

# Detect blue badge
hsv = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2HSV)
lower_blue = np.array([95, 80, 80], dtype=np.uint8)
upper_blue = np.array([135, 255, 255], dtype=np.uint8)
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
blue_ratio = cv2.countNonZero(blue_mask) / max(blue_mask.size, 1)

print(f"  Blue ratio: {blue_ratio:.3f}")

# Isolate white text
gray_badge = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2GRAY)
_, text_mask = cv2.threshold(gray_badge, 200, 255, cv2.THRESH_BINARY)
text_only = cv2.bitwise_and(text_mask, blue_mask)

print(f"  Text mask before upscale: {text_only.shape}, non-zero: {cv2.countNonZero(text_only)}")

# Upscale
scale = max(2, 120 // max(text_only.shape[1], 1))
print(f"  Scale factor: {scale}")

if scale > 1:
    text_upscaled = cv2.resize(
        text_only,
        (text_only.shape[1] * scale, text_only.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
else:
    text_upscaled = text_only

print(f"  Text mask after upscale: {text_upscaled.shape}, non-zero: {cv2.countNonZero(text_upscaled)}")

# Save debug images
debug_dir = Path("B:/Projects/schale/debug_ocr")
debug_dir.mkdir(exist_ok=True)

cv2.imwrite(str(debug_dir / "01_badge_roi.png"), badge_roi)
cv2.imwrite(str(debug_dir / "02_blue_mask.png"), blue_mask)
cv2.imwrite(str(debug_dir / "03_text_mask.png"), text_mask)
cv2.imwrite(str(debug_dir / "04_text_only.png"), text_only)
cv2.imwrite(str(debug_dir / "05_text_upscaled.png"), text_upscaled)

print(f"\n  Debug images saved to {debug_dir}")

# Try OCR
try:
    from schale.scanner._ocr import _run_ocr
    texts = _run_ocr(text_upscaled, allowlist="T0123456789")
    print(f"  OCR results: {texts}")
except Exception as e:
    print(f"  OCR error: {e}")
