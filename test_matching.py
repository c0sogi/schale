"""Test template matching on a single cell."""

import cv2
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid
from schale.scanner._icons import IconAtlas, TARGET_ICON_SIZE
from schale.scanner import _extract_icon_region

print("Loading image and detecting grid...")
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)
print(f"Detected {len(cells)} cells")

print("\nPreparing icon atlas...")
atlas = IconAtlas()
atlas.prepare()
print(f"Atlas has {len(atlas._templates)} templates")

# Test first cell
print(f"\nTesting cell at row=0, col=0...")
cell = cells[0]
cell_roi = cell.extract_roi(img)
print(f"Cell ROI shape: {cell_roi.shape}")

# Extract icon region
icon_roi = _extract_icon_region(cell_roi)
print(f"Icon ROI shape: {icon_roi.shape}")
print(f"Expected shape: {TARGET_ICON_SIZE}")

# Try matching without any filters
best_match, confidence = atlas.match(icon_roi, filter_blueprint=None, filter_tier=None)

print(f"\nMatching result:")
print(f"  Best match: {best_match.icon_name if best_match else 'None'}")
print(f"  Confidence: {confidence:.4f}")
print(f"  Default threshold: 0.45")

if best_match:
    print(f"  Matched item: T{best_match.tier} {best_match.category}")
else:
    print(f"  No match found (confidence too low)")

# Test multiple cells to see the pattern
print(f"\nTesting first 5 cells:")
for i in range(min(5, len(cells))):
    cell = cells[i]
    cell_roi = cell.extract_roi(img)
    icon_roi = _extract_icon_region(cell_roi)
    best_match, conf = atlas.match(icon_roi)

    match_str = f"{best_match.icon_name}" if best_match else "NO_MATCH"
    print(f"  Cell ({cell.row},{cell.col}): {match_str} (conf={conf:.4f})")
