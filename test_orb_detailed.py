"""Detailed ORB matching test to see what each cell matches."""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid, CellRegion
from schale.scanner._icons import IconAtlas
import cv2

# Load and normalize
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
original_h, original_w = img.shape[:2]

# Detect grid
cells_pass1 = detect_grid(img)
widths = [c.width for c in cells_pass1]
heights = [c.height for c in cells_pass1]
median_width = sorted(widths)[len(widths) // 2]
median_height = sorted(heights)[len(heights) // 2]

# Normalize
TEMPLATE_WIDTH = 146
TEMPLATE_HEIGHT = 116
scale_w = TEMPLATE_WIDTH / median_width
scale_h = TEMPLATE_HEIGHT / median_height

new_width = int(original_w * scale_w)
new_height = int(original_h * scale_h)
img_normalized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Scale cells
cells = [
    CellRegion(
        row=c.row, col=c.col,
        x=int(c.x * scale_w),
        y=int(c.y * scale_h),
        width=int(c.width * scale_w),
        height=int(c.height * scale_h),
    )
    for c in cells_pass1
]

# Prepare atlas
atlas = IconAtlas()
atlas.prepare()

print(f"Testing {len(cells)} cells with ORB feature matching:\n")

for cell in cells:
    cell_roi = cell.extract_roi(img_normalized)
    template, confidence = atlas.match(cell_roi)

    status = "MATCHED" if template else "UNRECOGNIZED"
    print(f"Cell ({cell.row},{cell.col}): {status}", end="")

    if template:
        print(f" -> {template.icon_name} T{template.tier} (conf: {confidence:.3f})")
    else:
        print()
