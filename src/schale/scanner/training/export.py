"""ONNX export utility for converting trained PyTorch models to ONNX format.

This module provides functionality to export trained EquipmentClassifier models to ONNX
format for efficient inference in production. ONNX models are:
- Framework-agnostic (can run without PyTorch)
- Optimized for inference (constant folding, operator fusion)
- Smaller in size (no training-specific overhead)
- Portable across different runtimes (ONNX Runtime, TensorRT, etc.)

The export process includes:
1. Loading trained checkpoint
2. Converting to ONNX with dynamic batch size support
3. Verification that ONNX output matches PyTorch output
4. Validation of model metadata and dimensions

Usage:
    # Export best checkpoint to default location
    python -m schale.scanner.training.export checkpoints/best_model.pth

    # Export to custom location
    python -m schale.scanner.training.export checkpoints/best_model.pth --output custom.onnx

    # Specify ONNX opset version
    python -m schale.scanner.training.export checkpoints/best_model.pth --opset 18
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from schale.scanner.training.model import create_model

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path | None = None,
    opset_version: int = 17,
    num_classes: int = 191,
    verify: bool = True,
) -> None:
    """Export trained PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to trained model checkpoint (.pth file)
        output_path: Path for output ONNX file. Defaults to
            src/schale/scanner/models/equipment_classifier.onnx
        opset_version: ONNX opset version (default: 17, supports PyTorch 2.0+)
        num_classes: Number of equipment classes (default: 191)
        verify: Whether to verify ONNX model matches PyTorch output (default: True)

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If ONNX export or verification fails
        ValueError: If ONNX output does not match PyTorch output

    Example:
        >>> from pathlib import Path
        >>> checkpoint = Path("checkpoints/best_model.pth")
        >>> export_to_onnx(checkpoint)
        Successfully exported to src/schale/scanner/models/equipment_classifier.onnx
        Model size: 5.43 MB
    """
    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Set default output path
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent / "models" / "equipment_classifier.onnx"
        )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)

    # Load model architecture
    model = create_model(num_classes=num_classes, pretrained=False)

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with optimizer state, metrics, etc.
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        val_acc = checkpoint.get("val_acc", None)
        logger.info("Loaded checkpoint from epoch %s", epoch)
        if val_acc is not None:
            logger.info("Validation accuracy: %.2f%%", val_acc * 100)
    else:
        # Raw state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Model loaded successfully, starting ONNX export")

    # Create dummy input (batch_size=1, channels=3, height=116, width=146)
    # Note: Input shape is (C, H, W) = (3, 116, 146) as defined in model
    dummy_input = torch.randn(1, 3, 116, 146)

    # Export to ONNX with dynamic batch size support
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,  # Export trained weights
            opset_version=opset_version,
            do_constant_folding=True,  # Optimize by folding constants
            input_names=["input"],  # Input tensor name
            output_names=["output"],  # Output tensor name
            dynamic_axes={
                "input": {0: "batch_size"},  # Allow variable batch size
                "output": {0: "batch_size"},
            },
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    logger.info("ONNX export successful: %s", output_path)

    # Verify ONNX model if requested
    if verify:
        _verify_onnx_model(model, output_path, dummy_input, num_classes)

    # Print model statistics
    model_size_mb = output_path.stat().st_size / 1_000_000
    logger.info("Model size: %.2f MB", model_size_mb)
    logger.info("ONNX opset version: %d", opset_version)
    logger.info("Input shape: (batch_size, 3, 116, 146)")
    logger.info("Output shape: (batch_size, %d)", num_classes)


def _verify_onnx_model(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    test_input: torch.Tensor,
    num_classes: int,
) -> None:
    """Verify ONNX model produces same output as PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX file
        test_input: Test input tensor
        num_classes: Expected number of output classes

    Raises:
        RuntimeError: If ONNX model cannot be loaded
        ValueError: If ONNX output does not match PyTorch output

    Example:
        >>> model = create_model()
        >>> dummy_input = torch.randn(1, 3, 116, 146)
        >>> _verify_onnx_model(model, Path("model.onnx"), dummy_input, 191)
        ONNX verification passed: outputs match within tolerance
    """
    logger.info("Verifying ONNX model...")

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # Load ONNX model
    try:
        ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    # Run ONNX inference
    onnx_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    assert isinstance(onnx_outputs, list) and len(onnx_outputs) > 0
    onnx_output = onnx_outputs[0]
    assert isinstance(onnx_output, np.ndarray)

    # Verify shapes
    expected_shape = (test_input.shape[0], num_classes)
    if onnx_output.shape != expected_shape:
        raise ValueError(
            f"ONNX output shape mismatch: expected {expected_shape}, got {onnx_output.shape}"
        )

    # Compare outputs (allow small numerical differences from fp32 precision)
    if not np.allclose(pytorch_output, onnx_output, rtol=1e-5, atol=1e-5):
        max_diff = np.abs(pytorch_output - onnx_output).max()
        raise ValueError(
            f"ONNX output does not match PyTorch output. Max difference: {max_diff:.6e}"
        )

    logger.info("ONNX verification passed: outputs match within tolerance")

    # Additional checks
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    logger.info("Max absolute difference: %.6e", max_diff)
    logger.info("Mean absolute difference: %.6e", mean_diff)

    # Verify class predictions are identical
    pytorch_class = np.argmax(pytorch_output, axis=1)
    onnx_class = np.argmax(onnx_output, axis=1)
    if not np.array_equal(pytorch_class, onnx_class):
        raise ValueError(
            f"ONNX predictions differ from PyTorch: "
            f"PyTorch={pytorch_class}, ONNX={onnx_class}"
        )

    logger.info("ONNX class predictions match PyTorch predictions")


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export best checkpoint
  python -m schale.scanner.training.export checkpoints/best_model.pth

  # Export to custom location
  python -m schale.scanner.training.export checkpoints/best_model.pth --output custom.onnx

  # Skip verification (faster, not recommended)
  python -m schale.scanner.training.export checkpoints/best_model.pth --no-verify

  # Use different ONNX opset version
  python -m schale.scanner.training.export checkpoints/best_model.pth --opset 18
        """,
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for ONNX file (default: src/schale/scanner/models/equipment_classifier.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, supports PyTorch 2.0+)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=191,
        help="Number of equipment classes (default: 191)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification (not recommended)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Export model
    try:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            opset_version=args.opset,
            num_classes=args.num_classes,
            verify=not args.no_verify,
        )
        logger.info("Export completed successfully!")
    except Exception as e:
        logger.error("Export failed: %s", e)
        raise


if __name__ == "__main__":
    main()
