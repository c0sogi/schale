"""Example usage of EquipmentDataset for training icon classification model.

This script demonstrates how to use the EquipmentDataset class to create
DataLoaders for training a PyTorch model on SchaleDB equipment icons.
"""

from typing import Sized, cast

from schale.scanner.training.dataset import EquipmentDataset, create_dataloaders


def example_basic_usage():
    """Example 1: Basic dataset creation and iteration."""
    print("=" * 60)
    print("Example 1: Basic Dataset Usage")
    print("=" * 60)

    # Create dataset with small augmentation factor for demo
    dataset = EquipmentDataset(
        augment_factor=10,  # Each icon generates 10 augmented samples
        split="train",
        seed=42,
        target_size=(116, 146),  # (H, W)
    )

    print("Dataset info:")
    print(f"  - Number of classes: {dataset.num_classes}")
    print(f"  - Number of base icons: {len(dataset.icons)}")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Augmentation factor: {dataset.augment_factor}")

    # Get a sample
    image_tensor, label = dataset[0]
    print("\nSample 0:")
    print(f"  - Image shape: {image_tensor.shape}")  # (C, H, W)
    print(f"  - Image dtype: {image_tensor.dtype}")  # float32
    print(f"  - Label: {label}")
    print(f"  - Label name: {dataset.class_to_name[str(label)]}")

    # Get multiple augmentations of the same icon
    print("\nAugmentations of icon 0 (first 3):")
    for i in range(3):
        img, lbl = dataset[i]
        print(f"  - Augmentation {i}: label={lbl}, shape={img.shape}")


def example_dataloader_usage():
    """Example 2: Create DataLoaders with train/val split."""
    print("\n" + "=" * 60)
    print("Example 2: DataLoader Creation")
    print("=" * 60)

    # Create train and validation DataLoaders
    train_loader, val_loader = create_dataloaders(
        batch_size=32,
        num_workers=0,  # Set to 4+ for actual training
        augment_factor=100,  # Recommended value for training
        train_split=0.8,
        seed=42,
    )

    print("DataLoader info:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Batch size: {train_loader.batch_size}")

    # Iterate through a few batches
    print("\nFirst 3 train batches:")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:
            break
        print(
            f"  - Batch {i}: images shape={images.shape}, labels shape={labels.shape}"
        )
        print(f"    Labels: {labels[:5].tolist()}...")


def example_training_loop():
    """Example 3: Basic training loop structure."""
    print("\n" + "=" * 60)
    print("Example 3: Training Loop Structure")
    print("=" * 60)

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        batch_size=16,
        num_workers=0,
        augment_factor=10,  # Small for demo
        train_split=0.8,
        seed=42,
    )

    # Simulate a training epoch
    print("\nSimulating training epoch...")
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        total_samples += len(labels)

        # Training code would go here:
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        if batch_idx % 20 == 0:
            print(
                f"  - Batch {batch_idx}/{len(train_loader)}: "
                f"processed {total_samples}/{len(cast(Sized, train_loader.dataset))} samples"
            )

    print(f"  - Epoch complete: {total_samples} samples processed")


def example_class_mapping():
    """Example 4: Working with class mappings."""
    print("\n" + "=" * 60)
    print("Example 4: Class Mapping Usage")
    print("=" * 60)

    dataset = EquipmentDataset(augment_factor=1, split="train")

    print(f"Total classes: {dataset.num_classes}")
    print("\nFirst 10 class mappings:")
    for i in range(10):
        class_name = dataset.class_to_name[str(i)]
        print(f"  - Class {i:3d}: {class_name}")

    # Reverse lookup
    print("\nReverse lookup examples:")
    example_names = [
        "equipment_icon_hat_tier5",
        "equipment_icon_gloves_tier10_piece",
        "equipment_icon_exp_3",
    ]
    for name in example_names:
        class_idx = dataset.name_to_class.get(name, -1)
        print(f"  - {name}: class {class_idx}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EquipmentDataset Usage Examples")
    print("=" * 70)

    example_basic_usage()
    example_dataloader_usage()
    example_training_loop()
    example_class_mapping()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
