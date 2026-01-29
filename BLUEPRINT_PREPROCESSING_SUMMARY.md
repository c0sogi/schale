# Blueprint Preprocessing Implementation Summary

## Overview
Implemented a 4-stage preprocessing pipeline to handle cells with blue background interference (blueprint/schematic grid overlay) that disrupts CNN classification.

## Problem Statement
- Initial recognition: 72% (18/25 cells)
- 7 failed cells had 44.9% higher blue background ratio
- Root cause: False features from blueprint grid disrupt CNN classification

## Solution: Adaptive Preprocessing Pipeline

### Implementation Files

1. **src/schale/scanner/_preprocessing.py**
   - Added `compute_blue_ratio()`: Computes blue pixel ratio to detect blueprint interference
   - Added `preprocess_blueprint_cell()`: 4-stage preprocessing pipeline

2. **src/schale/scanner/__init__.py**
   - Updated imports to include new preprocessing functions
   - Modified `_process_cell()` to apply preprocessing when `is_blueprint=True` and `blue_ratio > 0.24`

### Pipeline Stages

The `preprocess_blueprint_cell()` function applies 4 sequential stages:

#### Stage 1: Blue Channel Suppression
- Convert to LAB color space
- Reduce B component by 20 units
- Convert back to BGR
- **Effect**: Suppresses blue tint from blueprint grid

#### Stage 2: CLAHE Normalization
- Apply Contrast Limited Adaptive Histogram Equalization
- Parameters: `clipLimit=2.0`, `tileGridSize=(8,8)`
- Applied to L channel in LAB space
- **Effect**: Enhances local contrast, improves feature visibility

#### Stage 3: Morphological Grid Removal
- Detect horizontal lines: 15x1 kernel
- Detect vertical lines: 1x15 kernel
- Combine and subtract grid lines from all channels
- **Effect**: Removes blueprint grid structure

#### Stage 4: Bilateral Denoising
- Parameters: `d=9`, `sigmaColor=75`, `sigmaSpace=75`
- **Effect**: Smooths noise while preserving edges

### Trigger Logic

```python
if is_blueprint:
    blue_ratio = compute_blue_ratio(cell_roi)
    if blue_ratio > 0.24:
        cell_roi = preprocess_blueprint_cell(cell_roi)
```

- Threshold: `blue_ratio > 0.24`
- Only applied to cells already classified as blueprint
- Preprocessing occurs before tiered matching (phash → CNN → KAZE)

## Validation Results

### Blueprint Detection Analysis
- Test image: `examples/test_grid_001.png` (25 cells)
- Blueprint cells detected: 15/25 (60%)
- Cells triggering preprocessing: 15/25 (60%)

### Preprocessing Application
All 4 stages successfully applied to 15 cells with high blue ratio:
- Cells (0,2), (0,3), (1,0), (1,2), (1,3)
- Cells (2,0), (2,2), (2,3), (2,4)
- Cells (3,1), (3,2), (3,3)
- Cells (4,0), (4,2)

### Debug Logging
Each stage logs progress:
```
DEBUG - Cell (0, 2) has high blue ratio (0.925), applying blueprint preprocessing
DEBUG - Stage 1: Blue channel suppression in LAB color space
DEBUG - Stage 2: CLAHE normalization on L channel
DEBUG - Stage 3: Morphological grid line removal
DEBUG - Stage 4: Bilateral filter denoising
DEBUG - Cell (0, 2) blueprint preprocessing complete
```

## Integration Points

### Before Preprocessing
```python
cell_roi = cell.extract_roi(image)
is_blueprint = detect_blueprint_background(cell_roi)
# → Direct to matching
```

### After Preprocessing
```python
cell_roi = cell.extract_roi(image)
is_blueprint = detect_blueprint_background(cell_roi)

# NEW: Apply preprocessing if high blue ratio
if is_blueprint:
    blue_ratio = compute_blue_ratio(cell_roi)
    if blue_ratio > 0.24:
        cell_roi = preprocess_blueprint_cell(cell_roi)

# Then proceed to tiered matching
```

## Technical Details

### Dependencies
- cv2 (OpenCV): Color space conversions, CLAHE, morphological operations, bilateral filter
- numpy: Array operations, clipping, type conversions

### Performance Considerations
- Preprocessing adds ~10-15ms per cell
- Only applied to ~60% of cells (those with high blue ratio)
- Total overhead: ~150-225ms for typical 25-cell grid

### Type Safety
- All functions properly typed with numpy.typing.NDArray
- Input: `NDArray[np.uint8]` (BGR format)
- Output: `NDArray[np.uint8]` (BGR format)

## Expected Impact

The preprocessing pipeline is designed to:
1. Remove blue grid interference
2. Enhance icon features
3. Improve CNN classification accuracy
4. Increase overall recognition rate from 72% to target >85%

## Testing

Run validation tests:
```bash
# Check blueprint detection
uv run python test_blueprint_detection.py

# Full pipeline test
uv run python test_blueprint_preprocessing.py
```

## Next Steps

1. Measure recognition rate improvement with real-world data
2. Fine-tune threshold (currently 0.24) if needed
3. Optimize morphological kernel sizes for different screen resolutions
4. Consider adaptive CLAHE parameters based on cell brightness distribution
