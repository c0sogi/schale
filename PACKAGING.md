# 📦 Packaging Guide

## What's Included in the Wheel

The distributed wheel package (`schale-0.1.0-py3-none-any.whl`) includes:

### ✅ Ready-to-Use Assets (7.3 MB total)

1. **Pre-trained CNN Model** (~7 MB)
   - `schale/scanner/models/equipment_classifier.onnx` (296 KB)
   - `schale/scanner/models/equipment_classifier.onnx.data` (6.9 MB)
   - `schale/scanner/models/class_mapping.json` (17 KB)

2. **Bundled Equipment Icons** (1.6 MB)
   - 191 equipment icons in WebP format
   - Located in `schale/scanner/data/icons/`
   - Covers all equipment categories from SchaleDB

3. **Python Modules** (~500 KB)
   - Scanner implementation
   - Training pipeline (for reference)
   - Schema definitions
   - Cache control

**Total package size:** 7.3 MB (233 files)

## 🚀 Installation Experience

### For End Users (Production)
```bash
pip install schale[scanner]
```

**What happens:**
1. Downloads `schale-0.1.0-py3-none-any.whl` (7.3 MB)
2. Installs scanner dependencies (~300 MB): opencv, easyocr, onnxruntime
3. **No additional setup needed** - ready to use immediately!
4. No icon downloads on first run
5. No model downloads required

**First run:**
```python
from schale.scanner import scan_inventory

# Works immediately - no downloads!
result = scan_inventory("screenshot.png")
```

### For Developers
```bash
git clone <repo>
cd schale
uv sync  # Installs everything including PyTorch
```

## 📂 File Locations After Installation

### Installed Package Structure
```
site-packages/
└── schale/
    ├── scanner/
    │   ├── data/
    │   │   └── icons/              # 191 bundled icons ✅
    │   │       ├── equipment_icon_badge_tier1.webp
    │   │       ├── equipment_icon_badge_tier2.webp
    │   │       └── ...
    │   ├── models/
    │   │   ├── equipment_classifier.onnx      ✅
    │   │   ├── equipment_classifier.onnx.data ✅
    │   │   └── class_mapping.json             ✅
    │   ├── training/             # Training scripts (reference)
    │   ├── __init__.py           # Main scanner API
    │   ├── _cnn.py               # CNN inference
    │   ├── _icons.py             # Icon loading & matching
    │   ├── _grid.py              # Grid detection
    │   └── _ocr.py               # OCR utilities
    └── ...
```

### Runtime Cache (Optional Updates)
```
~/.schale/cache/
└── icons/                     # Downloaded updates (if any)
    └── equipment_icon_*.webp
```

## 🔄 Icon Loading Priority

The scanner checks for icons in this order:

1. **Bundled package data** (inside wheel) - FIRST ✅
   - `schale/scanner/data/icons/`
   - No network required
   - Always available

2. **User cache directory** - SECOND
   - `~/.schale/cache/icons/`
   - Used for manual updates
   - Can override bundled icons

3. **Download from SchaleDB** - FALLBACK
   - Only if icon not found in (1) or (2)
   - Rarely needed for bundled icons

## 📊 Size Breakdown

| Component | Size | Purpose |
|-----------|------|---------|
| CNN Model (ONNX) | 7.0 MB | Equipment classification |
| Equipment Icons | 1.6 MB | Template matching fallback |
| Python Code | 0.5 MB | Scanner implementation |
| Metadata | 0.2 MB | Package info, dependencies |
| **Total Wheel** | **7.3 MB** | Ready-to-use package |

## 🎯 Build Process

### Creating a Distribution
```bash
# Clean previous builds
rm -rf dist/ build/

# Build wheel
uv build --wheel

# Output: dist/schale-0.1.0-py3-none-any.whl
```

### Verifying Package Contents
```bash
# List all files in wheel
unzip -l dist/schale-0.1.0-py3-none-any.whl

# Check specific directories
unzip -l dist/*.whl | grep "icons/"
unzip -l dist/*.whl | grep "models/"

# Extract to inspect
unzip dist/schale-0.1.0-py3-none-any.whl -d /tmp/inspect
```

### Testing Installed Package
```bash
# Install in a clean environment
python -m venv test_env
source test_env/bin/activate  # or `test_env\Scripts\activate` on Windows

# Install wheel
pip install dist/schale-0.1.0-py3-none-any.whl[scanner]

# Test import and scan
python -c "
from schale.scanner import scan_inventory
result = scan_inventory('screenshot.png')
print(f'Found {len(result.items)} items')
"
```

## 📝 Configuration Files

### MANIFEST.in
```
# Include all data files in the package
recursive-include src/schale/scanner/data *.webp
recursive-include src/schale/scanner/models *.onnx *.json
include src/schale/scanner/models/*.data
```

**Purpose:** Ensures data files are included in source distributions (sdist).

**Note:** With `uv build` and src layout, most files are automatically included,
but MANIFEST.in provides explicit control and documentation.

### pyproject.toml
```toml
[tool.uv]
package = true
```

**Purpose:** Enables uv's build backend which automatically includes all files
in `src/` layout packages.

## 🔍 Debugging Package Issues

### Icon Not Found
```python
from schale.scanner._icons import _download_icon
path = _download_icon("equipment_icon_badge_tier1")
print(f"Icon path: {path}")
print(f"Exists: {path.exists() if path else False}")
```

### Model Not Found
```python
from schale.scanner._cnn import CNNClassifier
from pathlib import Path

# Should find bundled model
classifier = CNNClassifier()
print(f"Model loaded: {classifier.session is not None}")
```

### Check Package Contents
```python
from importlib.resources import files

# List bundled icons
icons_dir = files("schale.scanner.data") / "icons"
icons = list(icons_dir.iterdir())
print(f"Bundled icons: {len(icons)}")

# Check model files
models_dir = files("schale.scanner.models")
models = list(models_dir.iterdir())
print(f"Model files: {[m.name for m in models]}")
```

## 🚢 Publishing to PyPI

### Test PyPI (Recommended First)
```bash
# Build
uv build --wheel

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ schale[scanner]
```

### Production PyPI
```bash
# Build
uv build --wheel

# Upload to PyPI
twine upload dist/*

# Users can now install
pip install schale[scanner]
```

## ✅ Benefits of Ready-to-Use Package

1. **No setup required** - Works immediately after `pip install`
2. **No network dependency** - All assets bundled
3. **Consistent experience** - Same icons/models for everyone
4. **Fast startup** - No downloads on first run
5. **Offline capable** - Can run without internet
6. **Predictable behavior** - No version drift from downloads

## 📈 Future Improvements

1. **Model versioning**: Track model version in metadata
2. **Icon updates**: Optional command to update bundled icons
3. **Platform-specific wheels**: Optimize for CPU architectures
4. **Model variants**: Bundle multiple models (accuracy vs speed)
5. **Lazy loading**: Only load icons when needed (reduce memory)
