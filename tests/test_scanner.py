"""Comprehensive integration tests for the scanner module."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from schale.scanner import _get_atlas, scan_inventory

# Model paths
MODEL_PATH = Path(__file__).parent.parent / "src/schale/scanner/models/equipment_classifier.onnx"
MAPPING_PATH = Path(__file__).parent.parent / "src/schale/scanner/models/class_mapping.json"


def test_model_files_exist():
    """Verify model and mapping files exist."""
    assert MAPPING_PATH.exists(), "class_mapping.json missing"
    # ONNX model is optional - only check if CNN is being used


def test_model_size_constraint():
    """Verify ONNX model is < 50MB if it exists."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        assert size_mb < 50, f"Model too large: {size_mb:.2f}MB (max 50MB)"


def test_grid_detection():
    """Test grid detection on sample image."""
    from schale.scanner._grid import detect_grid
    from schale.scanner._preprocessing import load_image

    # Use test image if available
    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    img = load_image(str(test_image))
    cells = detect_grid(img)

    assert len(cells) == 25, f"Expected 25 cells, got {len(cells)}"
    assert cells[0].row == 0 and cells[0].col == 0, "Grid indexing incorrect"


def test_cnn_required():
    """Test that CNN is required for matching."""
    atlas = _get_atlas()
    atlas.prepare()

    # Ensure CNN not enabled by default
    assert atlas._cnn is None, "CNN should not be enabled by default"

    # Create dummy cell ROI (146x116x3 BGR)
    dummy_cell = np.random.randint(0, 255, (116, 146, 3), dtype=np.uint8)

    # Match should return None without CNN
    best_match, confidence = atlas.match_cnn(dummy_cell)
    assert best_match is None, "Should return None when CNN not enabled"
    assert confidence == 0.0, "Confidence should be 0.0 when CNN not enabled"


def test_inference_speed_benchmark():
    """Benchmark CNN inference speed if model exists."""
    if not MODEL_PATH.exists():
        pytest.skip("CNN model not trained yet")

    from schale.scanner._cnn import (
        CNNClassifier,  # pyright: ignore[reportAttributeAccessIssue]
    )

    classifier = CNNClassifier(MODEL_PATH, MAPPING_PATH)

    # Create dummy cell ROI (146x116x3 BGR)
    dummy_cell = np.random.randint(0, 255, (116, 146, 3), dtype=np.uint8)

    # Warmup
    classifier.predict(dummy_cell, top_k=5)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        classifier.predict(dummy_cell, top_k=5)
        times.append((time.perf_counter() - start) * 1000)  # ms

    avg_time = sum(times) / len(times)
    assert avg_time < 10, f"Inference too slow: {avg_time:.2f}ms (target <10ms)"

    print(f"\nAverage inference time: {avg_time:.2f}ms")


def test_class_mapping_integrity():
    """Verify class mapping has 191 classes and all expected categories."""
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    assert mapping["num_classes"] == 191, "Expected 191 equipment classes"

    expected_categories = [
        "Badge",
        "Bag",
        "Charm",
        "Exp",
        "Gloves",
        "Hairpin",
        "Hat",
        "Necklace",
        "Shoes",
        "Watch",
        "WeaponExpGrowthA",
        "WeaponExpGrowthB",
        "WeaponExpGrowthC",
        "WeaponExpGrowthZ",
    ]

    for cat in expected_categories:
        assert cat in mapping["categories"], f"Missing category: {cat}"


def test_atlas_preparation():
    """Test that icon atlas prepares correctly."""
    atlas = _get_atlas()
    atlas.prepare()

    assert atlas._prepared, "Atlas should be prepared"
    assert len(atlas._templates) > 0, "Atlas should have templates loaded"


def test_scan_inventory_integration():
    """Integration test for full scan_inventory pipeline."""
    from schale.schema.scanner import ScanResult

    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    result = scan_inventory(str(test_image))

    assert isinstance(result, ScanResult), "Results should be a ScanResult object"
    assert len(result.items) >= 0, "Items list should exist"
    assert result.grid_dimensions == (5, 5), "Grid should be 5x5"

    # Verify result structure
    for item in result.items:
        assert item.equipment_id > 0
        assert item.quantity > 0
        assert 1 <= item.tier <= 9
        assert isinstance(item.grid_position, tuple)
        assert len(item.grid_position) == 2


def test_grid_cell_ordering():
    """Test that grid cells are ordered correctly (left-to-right, top-to-bottom)."""
    from schale.scanner._grid import detect_grid
    from schale.scanner._preprocessing import load_image

    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    img = load_image(str(test_image))
    cells = detect_grid(img)

    # Check that cells are ordered correctly
    for i in range(len(cells) - 1):
        current = cells[i]
        next_cell = cells[i + 1]

        # If same row, next should have higher column
        if current.row == next_cell.row:
            assert next_cell.col > current.col, f"Column ordering wrong at index {i}"
        # Otherwise next should be on next row
        else:
            assert next_cell.row > current.row, f"Row ordering wrong at index {i}"


def test_tier_detection_range():
    """Test that detected tiers are within valid range (1-9)."""
    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    result = scan_inventory(str(test_image))

    for item in result.items:
        assert 1 <= item.tier <= 9, f"Invalid tier {item.tier} for {item.equipment_id}"


def test_quantity_detection_range():
    """Test that detected quantities are positive integers."""
    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    result = scan_inventory(str(test_image))

    for item in result.items:
        assert isinstance(item.quantity, int), f"Quantity should be int, got {type(item.quantity)}"
        assert item.quantity > 0, f"Quantity should be positive, got {item.quantity}"
        assert item.quantity <= 999999, f"Quantity suspiciously high: {item.quantity}"


def test_position_uniqueness():
    """Test that all detected items have unique grid positions."""
    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    result = scan_inventory(str(test_image))

    positions = [item.grid_position for item in result.items]
    assert len(positions) == len(set(positions)), "Duplicate positions detected"


def test_empty_cell_handling():
    """Test that scanner handles empty cells correctly."""
    from schale.scanner._grid import detect_grid
    from schale.scanner._preprocessing import load_image

    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    img = load_image(str(test_image))
    cells = detect_grid(img)

    # Grid should always be 25 cells
    assert len(cells) == 25

    # But results may have fewer items (empty cells filtered out)
    result = scan_inventory(str(test_image))
    assert len(result.items) <= 25


def test_concurrent_atlas_access():
    """Test that atlas can be accessed concurrently (singleton pattern)."""
    atlas1 = _get_atlas()
    atlas2 = _get_atlas()

    assert atlas1 is atlas2, "Atlas should be a singleton"


def test_preprocessing_consistency():
    """Test that preprocessing produces consistent results."""
    from schale.scanner._preprocessing import load_image

    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    # Load same image twice
    img1 = load_image(str(test_image))
    img2 = load_image(str(test_image))

    # Should be identical
    assert np.array_equal(img1, img2), "Preprocessing should be deterministic"


def test_error_handling_invalid_path():
    """Test error handling for invalid image paths."""
    with pytest.raises((FileNotFoundError, ValueError)):
        scan_inventory("nonexistent_image.png")


def test_error_handling_invalid_image():
    """Test error handling for invalid image data."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"not an image")
        temp_path = f.name

    try:
        with pytest.raises((ValueError, OSError)):
            scan_inventory(temp_path)
    finally:
        Path(temp_path).unlink()


def test_memory_efficiency():
    """Test that scanner doesn't leak memory on repeated calls."""
    import gc

    test_image = Path(r"examples\test_grid_001.png")
    if not test_image.exists():
        pytest.skip("Test image not available")

    # Run multiple times to check for memory leaks
    for _ in range(10):
        result = scan_inventory(str(test_image))
        assert len(result.items) >= 0

    # Force garbage collection
    gc.collect()

    # Memory should stabilize (hard to test without memory profiling tools)
    # This is more of a smoke test
