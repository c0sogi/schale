# Dependencies Guide

## 📦 Dependency Structure

This project uses a clean separation between production, optional, and development dependencies:

```toml
[project]
dependencies = [...]              # Core libraries only (lightweight)

[project.optional-dependencies]
scanner = [...]                   # Scanner runtime (opencv, onnxruntime)

[dependency-groups]
dev = [...]                       # Development + training (includes PyTorch)
```

## 🎯 Installation Scenarios

### 1. **Production/End-user Installation**
```bash
# Minimal installation - core libraries only
pip install schale

# With scanner functionality (adds ~300MB)
pip install schale[scanner]
```

**Size:**
- Core: ~10 MB (pydantic, requests)
- With scanner: ~300 MB (+ opencv, easyocr, onnxruntime)

**Use case:** End users who want to use the scanner for Blue Archive inventory recognition.

---

### 2. **Development Installation**
```bash
# Clone repository
git clone <repository-url>
cd schale

# Install all dev dependencies (includes training tools)
uv sync

# This automatically installs:
# - Core dependencies
# - Dev tools (pyright, ruff, pytest)
# - Scanner dependencies (opencv, easyocr, onnxruntime)
# - Training dependencies (torch, torchvision, albumentations)
```

**Size:** ~3-5 GB (includes PyTorch with CUDA)

**Use case:** Contributors who want to:
- Develop new features
- Run tests
- Train new models
- Debug issues

---

## 📚 Dependency Details

### Core Dependencies (always installed)
```toml
pydantic>=2.12.5      # Data validation
requests>=2.32.5      # HTTP client
```

### Scanner Optional Dependencies
```toml
opencv-python>=4.10.0    # Image processing
easyocr>=1.7.2          # OCR for tier badges
numpy>=1.26.0           # Numerical operations
onnxruntime>=1.23.2     # CNN inference (lightweight!)
```

**Why ONNX Runtime?**
- Small size: ~30 MB (vs PyTorch ~2 GB)
- CPU-optimized inference
- Cross-platform compatibility
- Production-ready performance

### Development Dependencies (dev group only)
```toml
# Development tools
pyright>=1.1.407      # Type checker
ruff>=0.14.10         # Linter/formatter
pytest>=9.0.2         # Testing framework

# Training dependencies (HEAVY - dev only!)
torch>=2.10.0         # Deep learning framework (~2 GB with CUDA)
torchvision>=0.25.0   # Vision utilities
albumentations>=2.0.8 # Image augmentation
tqdm>=4.67.1          # Progress bars
onnxscript>=0.5.7     # ONNX export utilities
```

**Why separate training deps?**
- PyTorch is **massive** (~2-5 GB with CUDA)
- Only needed for model training (one-time activity)
- ONNX model is already trained and exported
- Production inference uses ONNX Runtime (lightweight)

---

## 🔄 Workflow

### For End Users
```bash
# Install package
pip install schale[scanner]

# Use scanner
from schale.scanner import scan_inventory
result = scan_inventory("screenshot.png")
```

### For Developers
```bash
# Setup dev environment
uv sync

# Run tests
uv run pytest

# Train new model (uses PyTorch from dev group)
uv run python src/schale/scanner/training/train.py

# Export to ONNX (for production)
uv run python -c "from schale.scanner.training.export import export_to_onnx; ..."
```

---

## 🎓 PyTorch Index Configuration

PyTorch is installed from platform-specific indices:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cuda", marker = "sys_platform != 'darwin'" },
]
```

**Behavior:**
- macOS: CPU-only PyTorch (~1 GB)
- Windows/Linux: CUDA 12.8 PyTorch (~4 GB)

---

## 📊 Size Comparison

| Installation Type | Total Size | Key Libraries |
|-------------------|------------|---------------|
| Core only | ~10 MB | pydantic, requests |
| + scanner | ~300 MB | + opencv, easyocr, onnxruntime |
| + dev (full) | ~3-5 GB | + torch, torchvision (CUDA) |

---

## ✅ Best Practices

1. **Production deployments:** Use `pip install schale[scanner]`
   - Lightweight (~300 MB)
   - Fast installation
   - No training dependencies

2. **Development:** Use `uv sync`
   - Includes everything
   - Reproducible via uv.lock
   - Training capabilities

3. **CI/CD:** Use `pip install schale[scanner]` + `pip install pytest`
   - Testing without heavy training deps
   - Faster CI runs

4. **Docker:** Multi-stage build
   ```dockerfile
   # Build stage (with dev dependencies)
   FROM python:3.12 as builder
   RUN uv sync
   RUN uv run python train.py  # Train if needed

   # Production stage (minimal)
   FROM python:3.12-slim
   COPY --from=builder /app/models /app/models
   RUN pip install schale[scanner]
   ```

---

## 🚀 Future Improvements

1. **Separate training package:** `schale-training` as optional install
2. **Pre-built wheels:** Faster opencv/torch installation
3. **Model registry:** Download pre-trained models on-demand
4. **Lighter OCR:** Replace easyocr with lighter alternative (if possible)
