"""Check SIFT feature extraction vs ORB."""

import cv2
from schale.scanner._preprocessing import load_image
from schale.scanner._grid import detect_grid, CellRegion

# Load and normalize
img = load_image(r"C:\Users\cosogi\Downloads\제목 없음.png")
original_h, original_w = img.shape[:2]

cells_pass1 = detect_grid(img)
widths = [c.width for c in cells_pass1]
heights = [c.height for c in cells_pass1]
median_width = sorted(widths)[len(widths) // 2]
median_height = sorted(heights)[len(heights) // 2]

TEMPLATE_WIDTH = 146
TEMPLATE_HEIGHT = 116
scale_w = TEMPLATE_WIDTH / median_width
scale_h = TEMPLATE_HEIGHT / median_height

new_width = int(original_w * scale_w)
new_height = int(original_h * scale_h)
img_normalized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

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

# Compare ORB vs SIFT for first 5 cells
orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)

print("Comparing ORB vs SIFT feature extraction:\n")

for i in range(5):
    cell = cells[i]
    cell_roi = cell.extract_roi(img_normalized)
    h, w = cell_roi.shape[:2]

    # Extract icon region
    icon_h = int(h * 0.80)
    icon_w = int(w * 0.95)
    margin_x = (w - icon_w) // 2
    icon_roi = cell_roi[0:icon_h, margin_x:margin_x+icon_w]

    # Convert to grayscale
    if len(icon_roi.shape) == 3:
        gray_roi = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = icon_roi

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_roi = clahe.apply(gray_roi)

    # Extract with ORB
    orb_kp, orb_desc = orb.detectAndCompute(gray_roi, None)

    # Extract with SIFT
    sift_kp, sift_desc = sift.detectAndCompute(gray_roi, None)

    print(f"Cell ({cell.row},{cell.col}):")
    print(f"  ORB:  {len(orb_kp) if orb_kp else 0} keypoints, {len(orb_desc) if orb_desc is not None else 0} descriptors")
    print(f"  SIFT: {len(sift_kp) if sift_kp else 0} keypoints, {len(sift_desc) if sift_desc is not None else 0} descriptors")
