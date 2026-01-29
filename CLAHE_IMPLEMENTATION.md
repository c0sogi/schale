# CLAHE Normalization Implementation

## Summary
Implemented Quick Win 1: Added CLAHE (Contrast Limited Adaptive Histogram Equalization) normalization to CNN inference preprocessing to match the template preprocessing pipeline.

## Changes Made

### File: `src/schale/scanner/_cnn.py`

**Location**: `_preprocess()` method, lines 188-196

**Implementation**:
```python
# Apply CLAHE normalization to match template preprocessing in _icons.py
# Convert to grayscale for CLAHE
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_normalized = clahe.apply(gray)

# Convert back to BGR then to RGB
bgr_normalized = cv2.cvtColor(gray_normalized, cv2.COLOR_GRAY2BGR)
rgb = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2RGB)
```

**Position in Pipeline**:
1. Resize to expected dimensions
2. Crop to remove UI overlays
3. Normalize background color
4. **Apply CLAHE normalization** ← NEW STEP
5. Convert BGR to RGB
6. Apply ImageNet normalization
7. Transpose and batch

## Technical Details

### CLAHE Parameters
- **clipLimit**: 2.0 (matches template preprocessing)
- **tileGridSize**: (8, 8) (matches template preprocessing)

### Processing Steps
1. Convert BGR to grayscale
2. Apply CLAHE to grayscale image
3. Convert grayscale back to BGR
4. Convert BGR to RGB for model input

### Rationale
The template preprocessing in `_icons.py` (lines 282-283) applies CLAHE normalization to all template icons:
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_normalized = clahe.apply(gray)
```

By applying the same normalization to inference inputs, we ensure consistent preprocessing between templates and cell ROIs, which should improve matching accuracy.

## Verification

### Type Checking
```
pyright: 0 errors, 0 warnings
```

### Preprocessing Test
- Input shape: (116, 146, 3)
- Output shape: (1, 3, 66, 122) [after cropping]
- Output dtype: float32
- Output range: [-2.07, 2.64] [ImageNet normalized]

## Next Steps
Test on actual inventory screenshots to measure impact on recognition accuracy.
