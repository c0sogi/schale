"""Debug grid detection to see what's happening."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image, to_grayscale

image_path = r"C:\Users\cosogi\Downloads\제목 없음.png"
img = load_image(image_path)

gray = to_grayscale(img)
img_h, img_w = gray.shape[:2]
img_area = img_h * img_w

print(f"Image size: {img_w}x{img_h} (area: {img_area})")

# Scale blockSize with image width (range: 11-51, must be odd)
block_size = max(11, min(51, int(img_w / 40)))
if block_size % 2 == 0:  # Must be odd for adaptiveThreshold
    block_size += 1

# Scale kernel with image width (range: 3-5)
kernel_size = max(3, min(5, int(img_w / 200)))

print(f"Scaled parameters: blockSize={block_size}, kernelSize={kernel_size}")

# Adaptive threshold
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=block_size,
    C=-10,
)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")

# Filter by area and aspect ratio
# Scale min_area threshold based on image size
min_area_ratio = 0.001 if img_w < 1200 else 0.005
min_area = img_area * min_area_ratio
max_area = img_area * 0.10   # 10%

print(f"\nArea filters:")
print(f"  min_area_ratio: {min_area_ratio} ({min_area_ratio*100}%)")
print(f"  min_area: {min_area:.0f} px")
print(f"  max_area: {max_area:.0f} px")

candidates = []
rejected = {"too_small": 0, "too_large": 0, "bad_aspect": 0}

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    aspect_ratio = w / h if h > 0 else 0.0

    if area < min_area:
        rejected["too_small"] += 1
        continue
    if area > max_area:
        rejected["too_large"] += 1
        continue
    if not (0.65 <= aspect_ratio <= 1.35):
        rejected["bad_aspect"] += 1
        continue

    candidates.append((x, y, w, h, area, aspect_ratio))

print(f"\nCandidates after initial filtering: {len(candidates)}")
print(f"Rejected: {rejected}")

if candidates:
    print(f"\nCandidate details (first 10):")
    for i, (x, y, w, h, area, ar) in enumerate(candidates[:10]):
        print(f"  {i+1}. pos=({x},{y}) size={w}x{h} area={area} aspect={ar:.2f}")

    # Check median area filtering
    areas = sorted(c[4] for c in candidates)
    median_area = areas[len(areas) // 2]
    lower_area = median_area * 0.6
    upper_area = median_area * 1.4

    print(f"\nArea clustering:")
    print(f"  median_area: {median_area:.0f}")
    print(f"  range: {lower_area:.0f} - {upper_area:.0f}")

    passed_area_filter = sum(1 for c in candidates if lower_area <= c[4] <= upper_area)
    print(f"  passed: {passed_area_filter}/{len(candidates)}")
