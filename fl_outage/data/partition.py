import json
import os
from typing import Dict, List, Optional
import numpy as np


def compute_group_class_proportions(num_classes: int = 10, s: float = 0.8) -> np.ndarray:
    """Computes target class distributions p_A and p_B given separation parameter s.
    
    Formula:
        Let A = {0, ..., C/2 - 1}.
        p_A(c) = (1 + s) / C if c in A else (1 - s) / C
        p_B(c) = (1 - s) / C if c in A else (1 + s) / C
    """
    assert 0.0 <= s <= 1.0, f"Separation s must be in [0, 1], got {s}"
    half = num_classes // 2
    p_A = np.zeros(num_classes, dtype=np.float64)
    p_B = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if c < half:
            p_A[c] = (1.0 + s) / num_classes
            p_B[c] = (1.0 - s) / num_classes
        else:
            p_A[c] = (1.0 - s) / num_classes
            p_B[c] = (1.0 + s) / num_classes
    return p_A, p_B


def partition_dataset(
    targets: np.ndarray,
    num_classes: int = 10,
    s: float = 0.8,
    alpha: float = 1.0,
    num_clients_per_group: int = 10,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partitions dataset indices across 2 * num_clients_per_group clients into Group A and Group B.
    
    Group A: clients [0, num_clients_per_group - 1]
    Group B: clients [num_clients_per_group, 2 * num_clients_per_group - 1]
    
    Between-group separation is controlled by s:
        s = 0.0 -> IID control (identical class expectations)
        s = 0.8 -> Primary signal test (90% mass in primary classes)
        s = 1.0 -> Disjoint stress test (Group A only sees {0..4}, Group B only {5..9})
        
    Within-group heterogeneity is controlled by Dirichlet concentration alpha.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    total_clients = 2 * num_clients_per_group
    group_A_clients = list(range(num_clients_per_group))
    group_B_clients = list(range(num_clients_per_group, total_clients))

    client_indices: Dict[int, List[int]] = {i: [] for i in range(total_clients)}

    for c in range(num_classes):
        class_mask = np.where(targets == c)[0]
        n_class = len(class_mask)
        if n_class == 0:
            continue
        
        # Shuffle class indices deterministically
        shuffled_class_idx = rng.permutation(class_mask)

        in_primary_A = (c < num_classes // 2)
        # Proportion of class c allocated to Group A
        frac_A = (1.0 + s) / 2.0 if in_primary_A else (1.0 - s) / 2.0
        n_A = int(round(n_class * frac_A))
        n_B = n_class - n_A

        indices_A = shuffled_class_idx[:n_A]
        indices_B = shuffled_class_idx[n_A:]

        # Allocate within Group A using Dirichlet
        if n_A > 0:
            props_A = rng.dirichlet(np.repeat(alpha, num_clients_per_group))
            counts_A = rng.multinomial(n_A, props_A)
            start = 0
            for client_id, count in zip(group_A_clients, counts_A):
                if count > 0:
                    client_indices[client_id].extend(indices_A[start : start + count].tolist())
                    start += count

        # Allocate within Group B using Dirichlet
        if n_B > 0:
            props_B = rng.dirichlet(np.repeat(alpha, num_clients_per_group))
            counts_B = rng.multinomial(n_B, props_B)
            start = 0
            for client_id, count in zip(group_B_clients, counts_B):
                if count > 0:
                    client_indices[client_id].extend(indices_B[start : start + count].tolist())
                    start += count

    # Shuffle each client's final indices deterministically
    for client_id in client_indices:
        rng.shuffle(client_indices[client_id])

    return client_indices


def save_partitions(partitions: Dict[int, List[int]], filepath: str):
    """Saves partitions to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({str(k): v for k, v in partitions.items()}, f, indent=2)


def load_partitions(filepath: str) -> Dict[int, List[int]]:
    """Loads partitions from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
