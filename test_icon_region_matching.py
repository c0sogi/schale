"""Test matching against icon region only vs full cell."""

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

# Test cell (0,4) which user says should be T10
cell = cells[4]
cell_roi = cell.extract_roi(img)
h, w = cell_roi.shape[:2]

print(f"Cell ({cell.row},{cell.col}) - Full size: {w}x{h}")

# Match against full cell
template_full, conf_full = atlas.match(cell_roi)
print(f"\nFull cell matching:")
if template_full:
    print(f"  {template_full.icon_name} T{template_full.tier} (confidence: {conf_full:.3f})")
else:
    print(f"  No match")

# Match against icon region only (top 75%)
icon_roi = cell_roi[0:int(h*0.75), :]
print(f"\nIcon region only ({icon_roi.shape[1]}x{icon_roi.shape[0]}) matching:")
template_icon, conf_icon = atlas.match(icon_roi)
if template_icon:
    print(f"  {template_icon.icon_name} T{template_icon.tier} (confidence: {conf_icon:.3f})")
else:
    print(f"  No match")

# Try a few more cells
print(f"\n--- Testing first 5 cells ---")
for i in range(5):
    cell = cells[i]
    cell_roi = cell.extract_roi(img)
    h, w = cell_roi.shape[:2]

    # Icon region only
    icon_roi = cell_roi[0:int(h*0.75), :]
    template, conf = atlas.match(icon_roi)

    print(f"Cell ({cell.row},{cell.col}): ", end="")
    if template:
        print(f"{template.icon_name} T{template.tier} (confidence: {conf:.3f})")
    else:
        print("No match")
