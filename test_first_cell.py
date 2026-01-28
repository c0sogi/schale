"""Debug first cell matching in detail."""

import cv2
import numpy as np
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid
from schale.scanner._icons import IconAtlas

print("Loading image and detecting grid...")
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)
print(f"Detected {len(cells)} cells")

print("\nPreparing icon atlas...")
atlas = IconAtlas()
atlas.prepare()
print(f"Atlas has {len(atlas._templates)} templates")

# Get first cell
cell = cells[0]
cell_roi = cell.extract_roi(img)
print(f"\nFirst cell at ({cell.row}, {cell.col}):")
print(f"  ROI shape: {cell_roi.shape}")

# Get top 10 matches
print("\nTesting match with full cell ROI...")
best_match, best_conf = atlas.match(cell_roi)

print(f"\nBest match: {best_match.icon_name if best_match else 'None'}")
print(f"Confidence: {best_conf:.4f}")

if best_match:
    print(f"  Category: {best_match.category}")
    print(f"  Tier: {best_match.tier}")
    print(f"  Blueprint: {best_match.is_blueprint}")

# Try to get top 5 matches manually
print("\nTop 5 candidates:")
scores = []
for name, tmpl in list(atlas._templates.items())[:50]:  # Test first 50 templates
    gray_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)

    if tmpl.grayscale.shape[0] > gray_roi.shape[0] or tmpl.grayscale.shape[1] > gray_roi.shape[1]:
        continue

    if tmpl.alpha_mask is not None:
        result = cv2.matchTemplate(gray_roi, tmpl.grayscale, cv2.TM_CCOEFF_NORMED, mask=tmpl.alpha_mask)
    else:
        result = cv2.matchTemplate(gray_roi, tmpl.grayscale, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, _ = cv2.minMaxLoc(result)
    scores.append((name, max_val, tmpl.category, tmpl.tier))

scores.sort(key=lambda x: x[1], reverse=True)
for i, (name, conf, cat, tier) in enumerate(scores[:5]):
    print(f"  {i+1}. {name} ({cat} T{tier}): {conf:.4f}")
