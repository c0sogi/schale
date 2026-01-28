"""Test matching by resizing icon region to exact template size."""

import cv2
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid
from schale.scanner._icons import IconAtlas

# Load image
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

# Prepare atlas
atlas = IconAtlas()
atlas.prepare()

# Template size
TEMPLATE_WIDTH = 146
TEMPLATE_HEIGHT = 116

print("Testing exact-size icon matching:\n")

for i in range(min(5, len(cells))):
    cell = cells[i]
    cell_roi = cell.extract_roi(img)
    h, w = cell_roi.shape[:2]

    # Extract icon region (top 70% of cell, centered horizontally)
    icon_h = int(h * 0.70)
    icon_w = int(w * 0.85)  # Leave some margin for borders
    margin_x = (w - icon_w) // 2

    icon_roi = cell_roi[0:icon_h, margin_x:margin_x+icon_w]

    # Resize to exact template size
    icon_resized = cv2.resize(icon_roi, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
                             interpolation=cv2.INTER_AREA)

    # Save for inspection
    cv2.imwrite(f"B:/Projects/schale/debug_matching/icon_resized_{cell.row}_{cell.col}.png",
               icon_resized)

    # Match
    template, conf = atlas.match(icon_resized)

    print(f"Cell ({cell.row},{cell.col}): ", end="")
    if template:
        print(f"{template.icon_name} T{template.tier} (conf: {conf:.3f})")
    else:
        print("No match")

print("\nSaved resized icon images to debug_matching/")
