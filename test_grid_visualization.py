"""Visualize grid detection on original image."""

import cv2
import numpy as np
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid

# Load original image
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
cells = detect_grid(img)

# Create visualization
vis_img = img.copy()

# Draw all cells with different colors per row
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

for cell in cells:
    color = colors[cell.row % len(colors)]

    # Draw rectangle
    cv2.rectangle(vis_img, (cell.x, cell.y),
                 (cell.x + cell.width, cell.y + cell.height),
                 color, 2)

    # Label with row,col
    label = f"({cell.row},{cell.col})"
    cv2.putText(vis_img, label, (cell.x + 5, cell.y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save visualization
cv2.imwrite("B:/Projects/schale/debug_matching/grid_visualization.png", vis_img)
print(f"Saved grid visualization with {len(cells)} cells")

# Print grid summary
rows = {}
for cell in cells:
    if cell.row not in rows:
        rows[cell.row] = []
    rows[cell.row].append(cell.col)

print("\nGrid structure:")
for row in sorted(rows.keys()):
    cols = sorted(rows[row])
    print(f"  Row {row}: {len(cols)} cells at columns {cols}")

# Check which cells are in the first row
first_row_cells = [c for c in cells if c.row == 0]
print(f"\nFirst row has {len(first_row_cells)} cells:")
for cell in sorted(first_row_cells, key=lambda c: c.col):
    print(f"  Col {cell.col}: position ({cell.x},{cell.y}) size {cell.width}x{cell.height}")
