"""Test tier OCR with upscaled image."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid
from schale.scanner._ocr import _run_ocr
import re

img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

cell = cells[4]  # (0, 4) - has T10
cell_roi = cell.extract_roi(img)

h, w = cell_roi.shape[:2]
badge_roi = cell_roi[int(h * 0.75) : h, 0 : int(w * 0.40)]

# Blue mask
hsv = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, np.array([95, 80, 80]), np.array([135, 255, 255]))

# Text mask with threshold 160
gray_badge = cv2.cvtColor(badge_roi, cv2.COLOR_BGR2GRAY)
_, text_mask = cv2.threshold(gray_badge, 160, 255, cv2.THRESH_BINARY)
text_only = cv2.bitwise_and(text_mask, blue_mask)

print(f"Text pixels before upscale: {cv2.countNonZero(text_only)}")

# Upscale
scale = max(2, 120 // max(text_only.shape[1], 1))
print(f"Scale factor: {scale}")

text_upscaled = cv2.resize(
    text_only,
    (text_only.shape[1] * scale, text_only.shape[0] * scale),
    interpolation=cv2.INTER_NEAREST,
)

print(f"Text pixels after upscale: {cv2.countNonZero(text_upscaled)}")
print(f"Upscaled size: {text_upscaled.shape}")

# Save upscaled image
cv2.imwrite("B:/Projects/schale/tier_upscaled.png", text_upscaled)
print(f"Saved upscaled image to tier_upscaled.png")

# Run OCR
print("\nRunning OCR...")
texts = _run_ocr(text_upscaled, allowlist="T0123456789")
print(f"OCR raw results: {texts}")

# Try pattern matching
TIER_PATTERN = re.compile(r"[Tt](\d{1,2})")
for text in texts:
    match = TIER_PATTERN.search(text)
    if match:
        print(f"  Matched: {match.group(0)} -> tier {match.group(1)}")
    else:
        print(f"  No match in: '{text}'")

# Try joining
joined = "".join(texts).upper().strip()
print(f"\nJoined text: '{joined}'")
match = TIER_PATTERN.search(joined)
if match:
    print(f"  Matched in joined: tier {match.group(1)}")
