"""Comprehensive training pipeline for equipment classifier.

This module provides a production-ready training pipeline with:
- Configurable hyperparameters via TrainConfig dataclass
- CrossEntropyLoss with label smoothing
- AdamW optimizer with cosine annealing scheduler and warmup
- Progress tracking with tqdm
- Checkpoint saving (best model + periodic)
- Early stopping with patience
- Training history logging to JSON

Example:
    >>> from schale.scanner.training.train import train, TrainConfig
    >>> config = TrainConfig(epochs=30, batch_size=32)
    >>> best_model_path = train(config)
    >>> print(f"Best model saved to: {best_model_path}")
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sized, cast

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from schale.scanner.training.dataset import create_dataloaders
from schale.scanner.training.model import count_parameters, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults.

    Attributes:
        epochs: Total number of training epochs (default: 30).
        batch_size: Batch size for training and validation (default: 32).
        lr: Initial learning rate (default: 1e-3).
        weight_decay: L2 regularization weight (default: 1e-4).
        augment_factor: Number of augmented samples per icon per epoch (default: 100).
        num_workers: Number of data loader worker processes (default: 0).
        device: Device to train on - "cuda" or "cpu" (auto-detected).
        checkpoint_dir: Directory to save model checkpoints (default: ./checkpoints).
        log_interval: Log training metrics every N batches (default: 10).
        save_interval: Save checkpoint every N epochs (default: 5).
        warmup_epochs: Number of warmup epochs (default: 5).
        patience: Early stopping patience (default: 10).
        label_smoothing: Label smoothing factor for loss (default: 0.1).
        seed: Random seed for reproducibility (default: 42).
        num_classes: Number of output classes (default: 191).
        target_size: Input image size as (height, width) (default: (116, 146)).
        train_split: Fraction of data for training (default: 0.8).
    """

    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    augment_factor: int = 100
    num_workers: int = 0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_interval: int = 10
    save_interval: int = 5
    warmup_epochs: int = 5
    patience: int = 10
    label_smoothing: float = 0.1
    seed: int = 42
    num_classes: int = 191
    target_size: tuple[int, int] = (116, 146)
    train_split: float = 0.8


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a cosine annealing scheduler with linear warmup.

    Learning rate increases linearly from 0 to initial_lr during warmup,
    then decreases following a cosine curve to 0.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler with warmup and cosine annealing.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    """Compute top-k accuracy for the specified values of k.

    Args:
        outputs: Model output logits of shape (batch_size, num_classes).
        targets: Ground truth labels of shape (batch_size,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        List of accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, batch_size)

        # Compare with targets
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.item() / batch_size * 100.0)

        return result


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> dict[str, float]:
    """Train the model for one epoch.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        epoch: Current epoch number (0-indexed).
        config: Training configuration.

    Returns:
        Dictionary with training metrics (loss, accuracy, top5_accuracy).
    """
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_correct_top5 = 0
    running_samples = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{config.epochs} [Train]",
        leave=False,
    )

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size

        # Compute accuracy
        acc1, acc5 = compute_accuracy(outputs, targets, topk=(1, 5))
        running_correct += acc1 / 100.0 * batch_size
        running_correct_top5 += acc5 / 100.0 * batch_size

        # Update progress bar
        if (batch_idx + 1) % config.log_interval == 0:
            current_loss = running_loss / running_samples
            current_acc = running_correct / running_samples * 100.0
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.2f}%",
                lr=f"{current_lr:.2e}",
            )

    # Compute epoch metrics
    epoch_loss = running_loss / running_samples
    epoch_acc = running_correct / running_samples * 100.0
    epoch_acc5 = running_correct_top5 / running_samples * 100.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "top5_accuracy": epoch_acc5,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> dict[str, float]:
    """Validate the model on the validation set.

    Args:
        model: The model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
        epoch: Current epoch number (0-indexed).
        config: Training configuration.

    Returns:
        Dictionary with validation metrics (loss, accuracy, top5_accuracy).
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_correct_top5 = 0
    running_samples = 0

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{config.epochs} [Val]",
        leave=False,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Compute metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size

        # Compute accuracy
        acc1, acc5 = compute_accuracy(outputs, targets, topk=(1, 5))
        running_correct += acc1 / 100.0 * batch_size
        running_correct_top5 += acc5 / 100.0 * batch_size

    # Compute epoch metrics
    epoch_loss = running_loss / running_samples
    epoch_acc = running_correct / running_samples * 100.0
    epoch_acc5 = running_correct_top5 / running_samples * 100.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "top5_accuracy": epoch_acc5,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    metrics: dict[str, float | dict[str, float]],
    path: Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        scheduler: The scheduler state.
        epoch: Current epoch number.
        metrics: Dictionary of metrics to save.
        path: Path to save the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
) -> dict[str, float | dict[str, float]]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.

    Returns:
        Dictionary with checkpoint metadata (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint["epoch"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
    }


def train(config: TrainConfig) -> Path:
    """Run the complete training pipeline.

    This function:
    1. Sets up reproducibility (seeds)
    2. Creates data loaders with train/val split
    3. Initializes model, optimizer, scheduler, and loss function
    4. Runs training loop with validation
    5. Saves checkpoints (best model + periodic)
    6. Implements early stopping
    7. Logs training history to JSON

    Args:
        config: Training configuration.

    Returns:
        Path to the best model checkpoint.

    Raises:
        RuntimeError: If training fails or no valid checkpoints are saved.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 70)
    logger.info("Equipment Classifier Training")
    logger.info("=" * 70)

    # Set random seeds for reproducibility
    set_seed(config.seed)
    logger.info("Random seed set to %d", config.seed)

    # Setup device
    device = torch.device(config.device)
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info(
            "CUDA memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints will be saved to: %s", config.checkpoint_dir.resolve())

    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment_factor=config.augment_factor,
            train_split=config.train_split,
            seed=config.seed,
            target_size=config.target_size,
        )
    except FileNotFoundError as e:
        logger.error("Failed to create data loaders: %s", e)
        raise RuntimeError(
            "Icon cache not found. Download icons first by calling:\n"
            "  from schale.scanner._icons import IconAtlas\n"
            "  IconAtlas.prepare()"
        ) from e

    logger.info("Train samples: %d", len(cast(Sized, train_loader.dataset)))
    logger.info("Val samples: %d", len(cast(Sized, val_loader.dataset)))
    logger.info("Train batches per epoch: %d", len(train_loader))
    logger.info("Val batches per epoch: %d", len(val_loader))

    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes=config.num_classes, pretrained=True)
    model = model.to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Model size: %.2f MB", (total_params * 4) / 1e6)

    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    logger.info(
        "Loss function: CrossEntropyLoss (label_smoothing=%.2f)", config.label_smoothing
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    logger.info(
        "Optimizer: AdamW (lr=%.2e, weight_decay=%.2e)", config.lr, config.weight_decay
    )

    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    # Create scheduler with warmup and cosine annealing
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Scheduler: Cosine annealing with warmup (%d warmup steps, %d total steps)",
        warmup_steps,
        total_steps,
    )

    # Training state
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_acc5": [],
        "val_loss": [],
        "val_acc": [],
        "val_acc5": [],
        "lr": [],
        "epoch": [],
    }

    best_model_path = config.checkpoint_dir / "best_model.pt"

    logger.info("=" * 70)
    logger.info("Starting training for %d epochs", config.epochs)
    logger.info("Early stopping patience: %d epochs", config.patience)
    logger.info("=" * 70)

    start_time = time.time()

    try:
        for epoch in range(config.epochs):
            epoch_start = time.time()

            # Training
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                config=config,
            )

            # Validation
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                config=config,
            )

            epoch_time = time.time() - epoch_start

            # Get current learning rate
            current_lr = float(scheduler.get_last_lr()[0])

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_acc5"].append(train_metrics["top5_accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_acc5"].append(val_metrics["top5_accuracy"])
            history["lr"].append(current_lr)
            history["epoch"].append(epoch + 1)

            # Log epoch summary
            logger.info(
                "Epoch %d/%d (%.1fs) - "
                "Train: loss=%.4f, acc=%.2f%% - "
                "Val: loss=%.4f, acc=%.2f%%, top5=%.2f%% - "
                "LR: %.2e",
                epoch + 1,
                config.epochs,
                epoch_time,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["top5_accuracy"],
                current_lr,
            )

            # Check for best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch + 1
                patience_counter = 0

                # Save best model
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={
                        "train": train_metrics,
                        "val": val_metrics,
                        "best_val_acc": best_val_acc,
                    },
                    path=best_model_path,
                )
                logger.info("New best model! Val accuracy: %.2f%%", best_val_acc)
            else:
                patience_counter += 1
                logger.info(
                    "No improvement for %d epoch(s). Best: %.2f%% at epoch %d",
                    patience_counter,
                    best_val_acc,
                    best_epoch,
                )

            # Save periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                periodic_path = (
                    config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={
                        "train": train_metrics,
                        "val": val_metrics,
                    },
                    path=periodic_path,
                )

            # Early stopping check
            if patience_counter >= config.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    config.patience,
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error("Training failed with error: %s", e)
        raise

    # Training complete
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("Total time: %.1f minutes", total_time / 60)
    logger.info(
        "Best validation accuracy: %.2f%% at epoch %d", best_val_acc, best_epoch
    )
    logger.info("Best model saved to: %s", best_model_path.resolve())
    logger.info("=" * 70)

    # Save training history
    history_path = config.checkpoint_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to: %s", history_path.resolve())

    # Verify best model exists
    if not best_model_path.exists():
        raise RuntimeError(
            "Best model checkpoint was not saved. Training may have failed."
        )

    return best_model_path


def main() -> None:
    """Main entry point for training script."""
    config = TrainConfig()

    print("\n" + "=" * 70)
    print("Equipment Classifier Training Configuration")
    print("=" * 70)
    print(f"  Epochs:         {config.epochs}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Learning rate:  {config.lr}")
    print(f"  Weight decay:   {config.weight_decay}")
    print(f"  Augment factor: {config.augment_factor}")
    print(f"  Device:         {config.device}")
    print(f"  Checkpoint dir: {config.checkpoint_dir.resolve()}")
    print(f"  Seed:           {config.seed}")
    print("=" * 70 + "\n")

    best_model_path = train(config)

    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
