"""CNN-based equipment classification using ONNX Runtime.

Provides a pre-trained ResNet18 model for classifying equipment icons
from Blue Archive inventory screenshots. Serves as an alternative to
template matching for improved accuracy and speed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import cv2
    import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
except ImportError as exc:
    raise ImportError(
        "The CNN classifier requires additional dependencies. "
        "Install them with: uv add schale[scanner]"
    ) from exc

logger = logging.getLogger(__name__)


class CNNClassifier:
    """ONNX Runtime wrapper for equipment classification.

    Uses a pre-trained ResNet18 model to classify equipment icons from
    cell ROIs extracted from inventory screenshots.

    Attributes:
        session: ONNX Runtime inference session.
        class_to_name: Mapping from class index to equipment icon name.
        num_classes: Total number of equipment classes.
        input_name: Name of the model's input tensor.
        output_name: Name of the model's output tensor.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Expected input dimensions (H, W)
    INPUT_HEIGHT = 116
    INPUT_WIDTH = 146

    def __init__(
        self,
        model_path: Path | None = None,
        mapping_path: Path | None = None,
    ) -> None:
        """Initialize the CNN classifier.

        Args:
            model_path: Path to the ONNX model file. Defaults to
                `models/equipment_classifier.onnx` relative to this module.
            mapping_path: Path to the class mapping JSON file. Defaults to
                `models/class_mapping.json` relative to this module.

        Raises:
            FileNotFoundError: If model or mapping file does not exist.
            RuntimeError: If ONNX Runtime fails to load the model.
        """
        # Resolve default paths
        module_dir = Path(__file__).parent
        if model_path is None:
            model_path = module_dir / "models" / "equipment_classifier.onnx"
        if mapping_path is None:
            mapping_path = module_dir / "models" / "class_mapping.json"

        # Validate paths
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Please ensure the model file is downloaded."
            )
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Class mapping not found at {mapping_path}. "
                "Please ensure the mapping file exists."
            )

        # Load class mapping
        logger.debug("Loading class mapping from %s", mapping_path)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)

        self.class_to_name: dict[str, str] = mapping["class_to_name"]
        self.num_classes: int = mapping["num_classes"]

        # Load ONNX model
        logger.debug("Loading ONNX model from %s", model_path)
        try:
            # Use CPU execution provider for compatibility
            # GPU providers can be added later: ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session: ort.InferenceSession = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX model from {model_path}: {exc}"
            ) from exc

        # Get input/output names from the model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        self.input_name: str = str(input_name)
        self.output_name: str = str(output_name)

        logger.info(
            "CNN classifier initialized: %d classes, input=%s, output=%s",
            self.num_classes,
            self.input_name,
            self.output_name,
        )

    def _preprocess(self, cell_roi: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess cell ROI for CNN inference.

        This preprocessing MUST match the validation transform used during training
        to ensure training-inference distribution alignment.

        Args:
            cell_roi: BGR (or grayscale) cell region of interest

        Returns:
            Preprocessed tensor of shape (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
        """
        # 1. Resize to model input size using INTER_AREA (matches training)
        resized = cv2.resize(
            cell_roi,
            (self.INPUT_WIDTH, self.INPUT_HEIGHT),  # 146, 116
            interpolation=cv2.INTER_AREA,
        )

        # 2. Convert BGR to RGB (training uses RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # 4. Apply ImageNet normalization (mean and std from training)
        normalized = (normalized - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # 5. Transpose to (C, H, W) format
        chw = np.transpose(normalized, (2, 0, 1))

        # 6. Add batch dimension
        return np.expand_dims(chw, axis=0)

    def predict(
        self,
        cell_roi: NDArray[np.uint8],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Predict the equipment class for a single cell ROI.

        Args:
            cell_roi: Input image in BGR format (H, W, 3), uint8.
            top_k: Number of top predictions to return.

        Returns:
            List of (icon_name, confidence) tuples, sorted by confidence
            in descending order. Confidence values are in [0, 1].

        Raises:
            ValueError: If top_k is invalid or exceeds num_classes.
        """
        if top_k < 1 or top_k > self.num_classes:
            raise ValueError(f"top_k must be in [1, {self.num_classes}], got {top_k}")

        # Preprocess
        input_tensor = self._preprocess(cell_roi)

        # Run inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            raise RuntimeError(f"ONNX inference failed: {exc}") from exc

        # Type assertions for ONNX Runtime output
        assert isinstance(outputs, list), f"Expected list, got {type(outputs)}"
        assert isinstance(outputs[0], np.ndarray), (
            f"Expected ndarray, got {type(outputs[0])}"
        )

        # Extract logits and apply softmax
        logits: NDArray[np.float32] = outputs[0][
            0
        ]  # Remove batch dimension: (num_classes,)
        exp_logits: NDArray[np.float32] = np.exp(
            logits - np.max(logits)
        )  # Numerical stability
        probabilities: NDArray[np.float32] = exp_logits / np.sum(exp_logits)

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = [
            (self.class_to_name[str(idx)], float(probabilities[idx]))
            for idx in top_indices
        ]

        return results

    def predict_batch(
        self,
        cell_rois: list[NDArray[np.uint8]],
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Predict equipment classes for a batch of cell ROIs.

        Args:
            cell_rois: List of input images in BGR format (H, W, 3), uint8.
            top_k: Number of top predictions to return per image.

        Returns:
            List of predictions for each input image. Each prediction is a
            list of (icon_name, confidence) tuples.

        Raises:
            ValueError: If top_k is invalid or exceeds num_classes.
        """
        if top_k < 1 or top_k > self.num_classes:
            raise ValueError(f"top_k must be in [1, {self.num_classes}], got {top_k}")

        if not cell_rois:
            return []

        # Batch preprocessing
        input_batch = np.concatenate(
            [self._preprocess(roi) for roi in cell_rois],
            axis=0,
        )

        # Run batch inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_batch},
            )
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            raise RuntimeError(f"ONNX batch inference failed: {exc}") from exc

        # Type assertions for ONNX Runtime output
        assert isinstance(outputs, list), f"Expected list, got {type(outputs)}"
        assert isinstance(outputs[0], np.ndarray), (
            f"Expected ndarray, got {type(outputs[0])}"
        )

        # Extract logits: (batch_size, num_classes)
        logits_batch: NDArray[np.float32] = outputs[0]

        # Apply softmax per sample
        exp_logits: NDArray[np.float32] = np.exp(
            logits_batch - np.max(logits_batch, axis=1, keepdims=True)
        )
        probabilities_batch: NDArray[np.float32] = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        # Get top-k predictions for each sample
        results: list[list[tuple[str, float]]] = []
        for probabilities in probabilities_batch:
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            sample_results = [
                (self.class_to_name[str(idx)], float(probabilities[idx]))
                for idx in top_indices
            ]
            results.append(sample_results)

        return results


__all__ = ["CNNClassifier"]
