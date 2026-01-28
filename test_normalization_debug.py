"""Debug template-based normalization process."""

import cv2
import numpy as np
from pathlib import Path
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid

# Load image
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
original_h, original_w = img.shape[:2]
print(f"Original image: {original_w}x{original_h}")

# First pass grid detection
print("\n=== FIRST PASS ===")
cells_pass1 = detect_grid(img)
print(f"Detected {len(cells_pass1)} cells")

if cells_pass1:
    widths = [c.width for c in cells_pass1]
    heights = [c.height for c in cells_pass1]
    median_width = sorted(widths)[len(widths) // 2]
    median_height = sorted(heights)[len(heights) // 2]
    print(f"Median cell size: {median_width}x{median_height}")
    print(f"Cell size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")

    # Calculate scale
    TEMPLATE_WIDTH = 146
    TEMPLATE_HEIGHT = 116
    scale_w = TEMPLATE_WIDTH / median_width
    scale_h = TEMPLATE_HEIGHT / median_height
    scale_factor = (scale_w + scale_h) / 2.0

    print(f"\nScale factor: {scale_factor:.3f}")
    print(f"  Template size: {TEMPLATE_WIDTH}x{TEMPLATE_HEIGHT}")
    print(f"  Scale_w: {scale_w:.3f}, Scale_h: {scale_h:.3f}")

    # Visualize first pass grid
    debug_img1 = img.copy()
    for cell in cells_pass1:
        cv2.rectangle(debug_img1, (cell.x, cell.y), (cell.x + cell.width, cell.y + cell.height), (0, 255, 0), 2)
        cv2.putText(debug_img1, f"({cell.row},{cell.col})", (cell.x + 5, cell.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    debug_dir = Path("B:/Projects/schale/debug_normalization")
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "01_original_grid.png"), debug_img1)
    print(f"\nSaved: {debug_dir / '01_original_grid.png'}")

    # Normalize
    if abs(scale_factor - 1.0) > 0.1:
        new_width = int(original_w * scale_factor)
        new_height = int(original_h * scale_factor)
        img_normalized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        print(f"\n=== AFTER NORMALIZATION ===")
        print(f"Normalized image: {new_width}x{new_height}")

        # Second pass grid detection
        cells_pass2 = detect_grid(img_normalized)
        print(f"Detected {len(cells_pass2)} cells")

        if cells_pass2:
            widths2 = [c.width for c in cells_pass2]
            heights2 = [c.height for c in cells_pass2]
            median_width2 = sorted(widths2)[len(widths2) // 2]
            median_height2 = sorted(heights2)[len(heights2) // 2]
            print(f"Median cell size: {median_width2}x{median_height2}")
            print(f"Cell size range: {min(widths2)}x{min(heights2)} to {max(widths2)}x{max(heights2)}")
            print(f"Target was: {TEMPLATE_WIDTH}x{TEMPLATE_HEIGHT}")

            # Visualize second pass grid
            debug_img2 = img_normalized.copy()
            for cell in cells_pass2:
                cv2.rectangle(debug_img2, (cell.x, cell.y), (cell.x + cell.width, cell.y + cell.height), (0, 0, 255), 2)
                cv2.putText(debug_img2, f"({cell.row},{cell.col})", (cell.x + 5, cell.y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imwrite(str(debug_dir / "02_normalized_grid.png"), debug_img2)
            print(f"\nSaved: {debug_dir / '02_normalized_grid.png'}")

            # Check which cells were lost
            coords_pass1 = {(c.row, c.col) for c in cells_pass1}
            coords_pass2 = {(c.row, c.col) for c in cells_pass2}
            lost_cells = coords_pass1 - coords_pass2
            new_cells = coords_pass2 - coords_pass1

            if lost_cells:
                print(f"\nLost cells after normalization: {sorted(lost_cells)}")
            if new_cells:
                print(f"New cells after normalization: {sorted(new_cells)}")

        print(f"\nDebug images saved to: {debug_dir}")
