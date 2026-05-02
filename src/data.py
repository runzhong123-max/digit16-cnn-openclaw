"""Data loading: MNIST → 16×16 grayscale with train/val/test split."""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int = 64,
    image_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    use_subset: bool = False,
    subset_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST, downsample to image_size×image_size, split train→train/val.

    Returns:
        train_loader, val_loader, test_loader
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 255] → [0.0, 1.0]
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean/std
        ]
    )

    # Full training set (official train split)
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Train/val split
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Debug subset mode
    if use_subset:
        train_set = _trim_subset(train_set, subset_size, seed)
        val_set = _trim_subset(val_set, max(subset_size // 8, 16), seed)
        test_set = _trim_subset(test_set, max(subset_size // 4, 32), seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def _trim_subset(dataset, size: int, seed: int) -> Subset:
    """Take a random subset of the given dataset."""
    n = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)
