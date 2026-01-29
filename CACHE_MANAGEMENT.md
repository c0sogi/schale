# 📦 Cache Management

## Overview

Schale bundles all required data (icons, models) directly in the package. **No automatic downloads occur during runtime.** This ensures:
- ✅ Offline capability
- ✅ Reproducible behavior
- ✅ No unexpected network calls
- ✅ Fast startup

## 🏗️ Architecture

### Before (Old Design)
```
Runtime:
1. Check ~/.schale/cache/icons/
2. If missing → Download from SchaleDB ❌ (automatic, unpredictable)
3. Cache for next time

Problems:
- Network required on first run
- Slow first startup
- Cache location varies by platform
- Automatic downloads may fail
```

### After (New Design)
```
Installation:
- Package includes 191 icons (1.6 MB)
- Package includes ONNX model (7 MB)
- Total: 7.3 MB ready-to-use ✅

Runtime:
1. Load from package data (always available)
2. No downloads, no network

Updates (explicit only):
$ schale cache update
```

## 📂 File Locations

### Package Structure
```
site-packages/schale/
├── scanner/
│   ├── data/
│   │   └── icons/              # 191 bundled icons
│   │       ├── equipment_icon_badge_tier1.webp
│   │       ├── equipment_icon_badge_tier2.webp
│   │       └── ...
│   ├── models/
│   │   ├── equipment_classifier.onnx
│   │   ├── equipment_classifier.onnx.data
│   │   └── class_mapping.json
│   └── ...
└── ...
```

### No External Cache
The old `~/.schale/cache/` directory is **no longer used**. All data is bundled in the package.

## 🔧 CLI Commands

### View Cache Information
```bash
$ schale cache info

Bundled icons: 191
Models: class_mapping.json, equipment_classifier.onnx, equipment_classifier.onnx.data
Total icon size: 1.1 MB
```

### Update Icons (Explicit Only)
```bash
# Update missing icons only
$ schale cache update

# Force re-download all icons
$ schale cache update --force
```

**Note:** This command updates icons **in the package data directory**. It modifies the installed package, so you may need write permissions.

### Scan Screenshot
```bash
# Basic scan
$ schale scan screenshot.png

# With custom confidence threshold
$ schale scan screenshot.png --confidence 0.05

# Disable CNN, use SIFT only
$ schale scan screenshot.png --no-cnn

# Save results to JSON
$ schale scan screenshot.png --output results.json
```

## 🐍 Python API

### Loading Icons (Automatic)
```python
from schale.scanner import scan_inventory

# Icons are loaded automatically from package data
result = scan_inventory("screenshot.png")
```

### Manual Icon Path (Advanced)
```python
from schale.scanner._icons import _get_package_icon_path

# Get path to bundled icon
path = _get_package_icon_path("equipment_icon_badge_tier1")
print(path)  # .../site-packages/schale/scanner/data/icons/equipment_icon_badge_tier1.webp
```

## 🔄 Update Process

### When to Update
Icon updates are **rarely needed**. Only update when:
- SchaleDB adds new equipment
- Icon files are corrupted
- You want to test unreleased equipment

### How It Works
```bash
$ schale cache update
```

1. Reads equipment database from SchaleDB JSON
2. Downloads missing icons to package data directory
3. Saves as `.webp` (or `.png` fallback)
4. Reports success/failure for each icon

### Update Output
```
Updating equipment icon cache...
  ✓ Downloaded: equipment_icon_new_item
  ✓ Downloaded: equipment_icon_another_new
  (Skipped 189 existing icons)

Summary:
  Downloaded: 2
  Skipped: 189
  Failed: 0
```

## ⚠️ Important Notes

### 1. No Automatic Downloads
```python
# ✅ This works immediately (uses bundled icons)
result = scan_inventory("screenshot.png")

# ❌ This does NOT trigger downloads if icon missing
# Instead, it logs a warning and returns None
from schale.scanner._icons import _download_icon
path = _download_icon("nonexistent_icon")  # Returns None
```

### 2. Package Modification
`schale cache update` modifies files in the installed package. This means:
- May require write permissions to `site-packages/`
- In virtual environments: usually works
- In system Python: may need `sudo` (not recommended)
- In conda environments: should work

### 3. Development vs Production

**Development (editable install):**
```bash
git clone <repo>
cd schale
uv sync

# Icons are in: src/schale/scanner/data/icons/
# Updates modify: src/schale/scanner/data/icons/
```

**Production (wheel install):**
```bash
pip install schale[scanner]

# Icons are in: site-packages/schale/scanner/data/icons/
# Updates modify: site-packages/schale/scanner/data/icons/
```

## 📊 Size Breakdown

| Component | Size | Count |
|-----------|------|-------|
| Equipment Icons | 1.6 MB | 191 files |
| ONNX Model | 7.0 MB | 1 file + data |
| Class Mapping | 17 KB | 1 file |
| **Total Package** | **7.3 MB** | **233 files** |

## 🚀 Benefits

### For End Users
- ✅ Works offline (no network required)
- ✅ Fast first run (no downloads)
- ✅ Reproducible (same icons for everyone)
- ✅ Predictable (no random download failures)
- ✅ Secure (no automatic network calls)

### For Developers
- ✅ Explicit updates (via CLI command)
- ✅ Version control friendly (icons in git)
- ✅ CI/CD friendly (no download steps)
- ✅ Testable (fixed dataset)

### For Distribution
- ✅ Self-contained package
- ✅ No post-install scripts
- ✅ Platform independent
- ✅ Airgap compatible

## 🔍 Troubleshooting

### Icon Not Found
```python
# Check if icon is bundled
from schale.scanner._icons import _get_package_icon_path
path = _get_package_icon_path("equipment_icon_mystery")

if path is None:
    print("Icon not bundled - run: schale cache update")
else:
    print(f"Icon found: {path}")
```

### Low Recognition Rate
```bash
# Try lower confidence threshold
$ schale scan screenshot.png --confidence 0.05
```

### Update Failed
```bash
# Check network connectivity
$ curl https://schaledb.com/images/equipment/icon/equipment_icon_badge_tier1.webp

# Try force update
$ schale cache update --force
```

### Permission Denied
```bash
# On system Python (not recommended)
$ sudo schale cache update

# Better: use virtual environment
$ python -m venv venv
$ source venv/bin/activate  # or `venv\Scripts\activate` on Windows
$ pip install schale[scanner]
$ schale cache update
```

## 🎯 Design Principles

1. **Explicit over Implicit**
   - Updates require explicit `schale cache update`
   - No silent background downloads

2. **Bundled over Downloaded**
   - Icons bundled in package
   - Download only when explicitly requested

3. **Offline First**
   - Works without network
   - Network only for updates

4. **Reproducible**
   - Same package → same behavior
   - No download race conditions

5. **Secure**
   - No automatic network calls
   - User controls when to download
