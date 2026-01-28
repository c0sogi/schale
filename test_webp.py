"""Test script to debug WebP loading issue."""

import cv2
import numpy as np
from pathlib import Path

# Test one WebP file
webp_path = Path(r"C:\Users\cosogi\.schale\cache\icons\equipment_icon_hat_tier1.webp")

if not webp_path.exists():
    print(f"File does not exist: {webp_path}")
    exit(1)

print(f"File exists: {webp_path}")
print(f"File size: {webp_path.stat().st_size} bytes")

# Test 1: Try old method (cv2.imread directly)
print("\nTest 1: cv2.imread(str(path))")
img1 = cv2.imread(str(webp_path), cv2.IMREAD_GRAYSCALE)
print(f"Result: {img1 is not None}")

# Test 2: Try new method (fromfile + imdecode)
print("\nTest 2: np.fromfile + cv2.imdecode")
try:
    img_buffer = np.fromfile(str(webp_path), dtype=np.uint8)
    print(f"Buffer loaded: {len(img_buffer)} bytes")
    img2 = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
    print(f"Result: {img2 is not None}")
    if img2 is None:
        # Check OpenCV WebP support
        build_info = cv2.getBuildInformation()
        if "WebP" in build_info:
            webp_line = [line for line in build_info.split("\n") if "WebP" in line]
            print(f"WebP info: {webp_line}")
        else:
            print("WebP not found in build info")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check if file is actually a valid WebP
print("\nTest 3: Check WebP signature")
with open(webp_path, "rb") as f:
    header = f.read(12)
    print(f"First 12 bytes: {header.hex()}")
    if header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
        print("Valid WebP signature")
    else:
        print("INVALID WebP signature - file may be corrupted")
