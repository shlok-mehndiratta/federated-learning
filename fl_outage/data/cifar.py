import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


# CIFAR-10 standard normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-100 standard normalization statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar10_transforms(augment: bool = True):
    """Returns torchvision transforms for CIFAR-10."""
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def load_cifar10(root: str = "./data/cache", download: bool = True) -> Tuple[Dataset, Dataset]:
    """Loads full CIFAR-10 train and test sets."""
    os.makedirs(root, exist_ok=True)
    train_transform = get_cifar10_transforms(augment=True)
    test_transform = get_cifar10_transforms(augment=False)

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
    return train_set, test_set


def get_cifar100_transforms(augment: bool = True):
    """Returns torchvision transforms for CIFAR-100."""
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])


def load_cifar100(root: str = "./data/cache", download: bool = True) -> Tuple[Dataset, Dataset]:
    """Loads full CIFAR-100 train and test sets."""
    os.makedirs(root, exist_ok=True)
    train_transform = get_cifar100_transforms(augment=True)
    test_transform = get_cifar100_transforms(augment=False)

    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_transform)
    return train_set, test_set



def create_deterministic_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """Creates a DataLoader with deterministic shuffling via torch.Generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )
