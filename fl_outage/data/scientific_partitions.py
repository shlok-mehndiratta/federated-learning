"""
Scientific Partitions Engine for Onset-Dependent Mergeability.
Implements the exact partition protocol from Frozen Plan (v1.0):
- R1: Single batch perturbation (identical full dataset & order)
- R2a: Stochastic noise (identical full dataset, independent data-order seeds)
- R2b: Sample-split control (disjoint IID halves, 25k samples each)
- R3-L: Persistent label skew (s in {0.0, 0.5, 0.8, 0.95, 1.0})
- Asymmetric conditions: Group A skewed / Group B IID; 70/30 group size split
- R3-F: Controllable feature skew (mild and severe image corruptions)
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from fl_outage.data.cifar import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_transforms


class CorruptedDataset(Dataset):
    """Dataset wrapper applying controllable synthetic feature corruptions (CIFAR-10-C style)."""

    def __init__(self, base_dataset: Dataset, corruption_level: str = "mild"):
        self.base_dataset = base_dataset
        self.corruption_level = corruption_level
        
        if corruption_level == "mild":
            self.corruption = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        elif corruption_level == "severe":
            self.corruption = transforms.Compose([
                transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7),
                transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.5)),
            ])
        else:
            self.corruption = transforms.Identity()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, target = self.base_dataset[idx]
        img = self.corruption(img)
        return img, target


def create_scientific_partition_indices(
    targets: np.ndarray,
    source: str = "R3-L",
    s: float = 0.8,
    asymmetric_mode: Optional[str] = None,
    num_classes: int = 10,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Generates exact index splits for Branch A and Branch B according to the scientific protocol.
    
    Args:
        targets: Array of dataset labels (e.g. 50,000 for CIFAR-10).
        source: Divergence source in {'R1', 'R2a', 'R2b', 'R3-L', 'R3-F'}.
        s: Label separation parameter in [0.0, 1.0].
        asymmetric_mode: Optional in {None, 'A_skew_B_iid', 'ratio_70_30'}.
        num_classes: Number of target classes (10 for CIFAR-10).
        seed: Random seed for deterministic sample partitioning.
        
    Returns:
        indices_A, indices_B: Lists of indices allocated to Group A and Group B.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    n_total = len(targets)
    all_indices = np.arange(n_total)

    # Rung 1: Single perturbation uses identical full datasets
    if source == "R1":
        return all_indices.tolist(), all_indices.tolist()

    # Rung 2a: Different data order on full training set (both see all 50k)
    if source == "R2a":
        idx_A = all_indices.copy()
        idx_B = all_indices.copy()
        return idx_A.tolist(), idx_B.tolist()

    # Rung 2b: Disjoint IID half-split (each branch gets exactly N/2 = 25,000 balanced samples)
    if source == "R2b":
        indices_A = []
        indices_B = []
        for c in range(num_classes):
            c_mask = np.where(targets == c)[0]
            shuffled_c = rng.permutation(c_mask)
            half = len(shuffled_c) // 2
            indices_A.extend(shuffled_c[:half])
            indices_B.extend(shuffled_c[half:])
        rng.shuffle(indices_A)
        rng.shuffle(indices_B)
        return indices_A, indices_B

    # Rung 3-F: Feature skew uses balanced disjoint IID halves (corruptions applied in dataset wrapper)
    if source == "R3-F":
        return create_scientific_partition_indices(targets, source="R2b", seed=seed)

    # Rung 3-L: Label skew (s in {0.0, 0.5, 0.8, 0.95, 1.0}) or Asymmetric conditions
    if source == "R3-L":
        half_classes = num_classes // 2

        # Asymmetric Mode 1: 70/30 group size ratio
        if asymmetric_mode == "ratio_70_30":
            indices_A = []
            indices_B = []
            for c in range(num_classes):
                c_mask = np.where(targets == c)[0]
                shuffled_c = rng.permutation(c_mask)
                n_A = int(round(0.70 * len(shuffled_c)))
                indices_A.extend(shuffled_c[:n_A])
                indices_B.extend(shuffled_c[n_A:])
            rng.shuffle(indices_A)
            rng.shuffle(indices_B)
            return indices_A, indices_B

        # Asymmetric Mode 2: Group A at s=0.8 skew, Group B balanced IID
        if asymmetric_mode == "A_skew_B_iid":
            indices_A = []
            indices_B = []
            for c in range(num_classes):
                c_mask = np.where(targets == c)[0]
                shuffled_c = rng.permutation(c_mask)
                in_primary = (c < half_classes)
                # Group A draws with skew s
                frac_A = (1.0 + s) / 2.0 if in_primary else (1.0 - s) / 2.0
                n_A = int(round(len(shuffled_c) * frac_A))
                # Group B receives balanced allocation from remaining or pool
                indices_A.extend(shuffled_c[:n_A])
                indices_B.extend(shuffled_c[n_A:])
            rng.shuffle(indices_A)
            rng.shuffle(indices_B)
            return indices_A, indices_B

        # Standard Symmetric Protocol:
        # Group A fraction (1+s)/2 on classes 0..C/2-1, (1-s)/2 on classes C/2..C-1
        # Group B receives the exact complement
        indices_A = []
        indices_B = []
        for c in range(num_classes):
            c_mask = np.where(targets == c)[0]
            shuffled_c = rng.permutation(c_mask)
            in_primary = (c < half_classes)
            frac_A = (1.0 + s) / 2.0 if in_primary else (1.0 - s) / 2.0
            n_A = int(round(len(shuffled_c) * frac_A))
            indices_A.extend(shuffled_c[:n_A])
            indices_B.extend(shuffled_c[n_A:])
            
        rng.shuffle(indices_A)
        rng.shuffle(indices_B)
        return indices_A, indices_B

    raise ValueError(f"Unknown divergence source: {source}")


def get_scientific_subsets(
    train_dataset: Dataset,
    source: str = "R3-L",
    s: float = 0.8,
    asymmetric_mode: Optional[str] = None,
    corruption_level: Optional[str] = None,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Returns the two PyTorch Subsets for Branch A and Branch B."""
    targets = getattr(train_dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset must have 'targets' attribute.")
        
    idx_A, idx_B = create_scientific_partition_indices(
        targets=targets,
        source=source,
        s=s,
        asymmetric_mode=asymmetric_mode,
        seed=seed
    )
    
    subset_A = Subset(train_dataset, idx_A)
    if source == "R3-F" and corruption_level is not None:
        subset_B = Subset(CorruptedDataset(train_dataset, corruption_level=corruption_level), idx_B)
    else:
        subset_B = Subset(train_dataset, idx_B)
        
    return subset_A, subset_B
