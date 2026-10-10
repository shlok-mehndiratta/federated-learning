import numpy as np
import pytest

from fl_outage.data.partition import partition_dataset, compute_group_class_proportions


def test_partition_s1_is_disjoint():
    """Gate G0 Test 3a: At s=1.0, Group A and Group B must have strictly disjoint classes."""
    num_classes = 10
    samples_per_class = 200
    targets = np.repeat(np.arange(num_classes), samples_per_class)

    partitions = partition_dataset(
        targets=targets,
        num_classes=num_classes,
        s=1.0,
        alpha=1.0,
        num_clients_per_group=10,
        seed=42
    )

    group_A_clients = list(range(10))
    group_B_clients = list(range(10, 20))

    group_A_targets = [targets[idx] for cid in group_A_clients for idx in partitions[cid]]
    group_B_targets = [targets[idx] for cid in group_B_clients for idx in partitions[cid]]

    unique_A = set(group_A_targets)
    unique_B = set(group_B_targets)

    # Group A must only see classes 0, 1, 2, 3, 4
    assert unique_A.issubset({0, 1, 2, 3, 4}), f"Group A saw classes outside 0..4: {unique_A}"
    # Group B must only see classes 5, 6, 7, 8, 9
    assert unique_B.issubset({5, 6, 7, 8, 9}), f"Group B saw classes outside 5..9: {unique_B}"
    # Intersection must be empty
    assert len(unique_A.intersection(unique_B)) == 0, "Group A and B overlap in classes at s=1.0!"


def test_partition_s0_is_balanced():
    """Gate G0 Test 3b: At s=0.0, both groups have balanced class expectations."""
    num_classes = 10
    samples_per_class = 500
    targets = np.repeat(np.arange(num_classes), samples_per_class)

    partitions = partition_dataset(
        targets=targets,
        num_classes=num_classes,
        s=0.0,
        alpha=1.0,
        num_clients_per_group=10,
        seed=42
    )

    group_A_clients = list(range(10))
    group_B_clients = list(range(10, 20))

    group_A_targets = [targets[idx] for cid in group_A_clients for idx in partitions[cid]]
    group_B_targets = [targets[idx] for cid in group_B_clients for idx in partitions[cid]]

    hist_A = np.bincount(group_A_targets, minlength=10)
    hist_B = np.bincount(group_B_targets, minlength=10)

    # Total samples should be equal
    assert len(group_A_targets) == len(group_B_targets)

    # Proportions of primary classes {0..4} should be approximately 50%
    prop_A_first_half = hist_A[:5].sum() / len(group_A_targets)
    prop_B_first_half = hist_B[:5].sum() / len(group_B_targets)

    assert prop_A_first_half == pytest.approx(0.5, abs=0.01)
    assert prop_B_first_half == pytest.approx(0.5, abs=0.01)


def test_no_data_leakage_or_duplicates():
    """Gate G0 Test 3c: All dataset indices are partitioned without duplication."""
    targets = np.repeat(np.arange(10), 100)
    partitions = partition_dataset(
        targets=targets,
        num_classes=10,
        s=0.8,
        alpha=0.5,
        num_clients_per_group=5,
        seed=123
    )

    all_indices = []
    for cid, idxs in partitions.items():
        all_indices.extend(idxs)

    assert len(all_indices) == len(targets), "Number of partitioned indices does not match dataset size"
    assert len(set(all_indices)) == len(targets), "Duplicate indices found across clients!"
