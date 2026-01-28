"""Debug template matching visually."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid, CellRegion
from schale.scanner._icons import IconAtlas

# Load and normalize image
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
scale_factor = (scale_w + scale_h) / 2.0

print(f"Original: {original_w}x{original_h}")
print(f"Scale factor: {scale_factor:.3f}")

if abs(scale_factor - 1.0) > 0.1:
    new_width = int(original_w * scale_factor)
    new_height = int(original_h * scale_factor)
    img_normalized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Scale cell coordinates
    cells = [
        CellRegion(
            row=c.row, col=c.col,
            x=int(c.x * scale_factor),
            y=int(c.y * scale_factor),
            width=int(c.width * scale_factor),
            height=int(c.height * scale_factor),
        )
        for c in cells_pass1
    ]
    print(f"Normalized: {new_width}x{new_height}")
else:
    img_normalized = img
    cells = cells_pass1

# Prepare atlas
atlas = IconAtlas()
atlas.prepare()

# Test first few cells
debug_dir = Path("B:/Projects/schale/debug_matching")
debug_dir.mkdir(exist_ok=True)

print(f"\nTesting first 5 cells:")
for i, cell in enumerate(cells[:5]):
    cell_roi = cell.extract_roi(img_normalized)
    print(f"\nCell ({cell.row},{cell.col}): size {cell_roi.shape[1]}x{cell_roi.shape[0]}")

    # Save cell image
    cv2.imwrite(str(debug_dir / f"cell_{cell.row}_{cell.col}.png"), cell_roi)

    # Try matching
    template, confidence = atlas.match(cell_roi)
    if template:
        print(f"  Matched: {template.icon_name} (confidence: {confidence:.3f})")

        # Load and save the matched template for comparison
        from pathlib import Path as P
        cache_dir = P.home() / '.schale' / 'cache' / 'icons'
        template_path = cache_dir / f"{template.icon_name}.webp"
        if template_path.exists():
            tmpl_buffer = np.fromfile(str(template_path), dtype=np.uint8)
            tmpl_img = cv2.imdecode(tmpl_buffer, cv2.IMREAD_COLOR)
            cv2.imwrite(str(debug_dir / f"template_{cell.row}_{cell.col}_{template.icon_name}.png"), tmpl_img)
    else:
        print(f"  No match")

print(f"\nDebug images saved to: {debug_dir}")
