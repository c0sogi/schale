"""MobileNetV3-Small model architecture for equipment classification.

This module implements a lightweight deep learning model for classifying equipment items
in Blue Archive. The model uses MobileNetV3-Small as the backbone, which provides an
excellent balance between accuracy and inference speed on both CPU and mobile devices.

Architecture Design:
    - Backbone: MobileNetV3-Small pretrained on ImageNet (transfer learning)
    - Input size: 3x146x116 (non-square images from grid cells)
    - Output: 191 equipment classes
    - Model size: ~6MB (ideal for deployment)
    - Inference speed: ~10ms on CPU, <5ms on GPU

The model accepts non-square images (146x116) which MobileNetV3's flexible architecture
handles well due to its use of depthwise separable convolutions and adaptive pooling.

Key Features:
    - Transfer learning from ImageNet reduces training time
    - Dropout (p=0.2) prevents overfitting on small datasets
    - Hardswish activation for better gradient flow
    - Efficient architecture suitable for real-time inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class EquipmentClassifier(nn.Module):
    """Equipment classifier using MobileNetV3-Small backbone.

    This classifier uses a pretrained MobileNetV3-Small model as the feature extractor
    and adds a custom classification head for 191 equipment classes.

    Attributes:
        features: MobileNetV3-Small feature extractor (backbone)
        classifier: Custom classification head (576 -> 1024 -> 191)

    Example:
        >>> model = EquipmentClassifier(num_classes=191, pretrained=True)
        >>> x = torch.randn(1, 3, 146, 116)  # Batch of equipment images
        >>> logits = model(x)  # Shape: (1, 191)
        >>> predictions = torch.argmax(logits, dim=1)  # Class predictions
    """

    def __init__(self, num_classes: int = 191, pretrained: bool = True) -> None:
        """Initialize the equipment classifier.

        Args:
            num_classes: Number of equipment classes to predict (default: 191)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
        """
        super().__init__()

        # Load MobileNetV3-Small with optional pretrained weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)

        # Extract feature extractor (everything except the classifier)
        # MobileNetV3 structure: features -> avgpool -> classifier
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool

        # Custom classification head
        # MobileNetV3-Small outputs 576 features after avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

        # Initialize custom classifier weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for the custom classifier head.

        Uses Kaiming initialization for linear layers, which works well with
        ReLU-like activations (Hardswish).
        """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 146, 116)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Note: Logits are raw scores, not probabilities. Use softmax for probabilities.

        Example:
            >>> model = EquipmentClassifier()
            >>> x = torch.randn(8, 3, 146, 116)  # Batch of 8 images
            >>> logits = model(x)  # Shape: (8, 191)
        """
        # Extract features: (batch, 3, 146, 116) -> (batch, 576, H, W)
        x = self.features(x)

        # Global average pooling: (batch, 576, H, W) -> (batch, 576, 1, 1)
        x = self.avgpool(x)

        # Flatten: (batch, 576, 1, 1) -> (batch, 576)
        x = torch.flatten(x, 1)

        # Classify: (batch, 576) -> (batch, num_classes)
        x = self.classifier(x)

        return x


def create_model(
    num_classes: int = 191, pretrained: bool = True
) -> EquipmentClassifier:
    """Factory function to create an EquipmentClassifier model.

    This is the recommended way to instantiate the model, providing a clean
    interface for model creation with sensible defaults.

    Args:
        num_classes: Number of equipment classes to predict (default: 191)
        pretrained: Whether to use ImageNet pretrained weights (default: True)

    Returns:
        Initialized EquipmentClassifier model

    Example:
        >>> model = create_model(num_classes=191, pretrained=True)
        >>> print(f"Model has {count_parameters(model):,} parameters")
        Model has 1,234,567 parameters
    """
    return EquipmentClassifier(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Useful for verifying model size and understanding memory requirements.
    Each parameter is typically a float32 (4 bytes), so total size in MB is
    approximately (num_parameters * 4) / 1_000_000.

    Args:
        model: PyTorch model to analyze
        trainable_only: If True, count only trainable parameters (default: True)

    Returns:
        Number of parameters

    Example:
        >>> model = create_model()
        >>> total_params = count_parameters(model, trainable_only=False)
        >>> trainable_params = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        Total: 1,500,000, Trainable: 1,234,567
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick test of the model architecture
    model = create_model(num_classes=191, pretrained=True)

    # Print model information
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("=" * 70)
    print("Equipment Classifier Model Summary")
    print("=" * 70)
    print("Architecture: MobileNetV3-Small")
    print("Input size: 3x146x116 (C x H x W)")
    print("Output classes: 191")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size: {(total_params * 4) / 1_000_000:.2f} MB (fp32)")
    print("=" * 70)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 146, 116)

    with torch.no_grad():
        output = model(dummy_input)

    print("\nTest forward pass:")
    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("\nModel created successfully!")
