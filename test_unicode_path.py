"""Test script to verify Unicode path handling fix."""

from schale.scanner._preprocessing import load_image

# Test with Korean filename
image_path = r"C:\Users\cosogi\Downloads\제목 없음.png"

try:
    img = load_image(image_path)
    print("SUCCESS: Loaded image with Unicode path")
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")
except FileNotFoundError:
    print("FAIL: File not found (check if the file exists)")
except ValueError as e:
    print(f"FAIL: Cannot decode image - {e}")
except Exception as e:
    print(f"FAIL: Unexpected error - {e}")
