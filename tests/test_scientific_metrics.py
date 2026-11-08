"""
Automated unit tests for scientific metrics and partitioning engine.
Verifies:
1. Identical models yield zero barrier and zero D_pair.
2. Symmetric pure logit-offset pair cancels out under averaging.
3. Angular CKA and unbiased HSIC properties (identity=1.0, orthogonal=0.0).
4. Cumulative partial merge consistency across stages.
5. Exact scientific partition invariants (R2b balanced half-split, R3-L disjointness, asymmetric 70/30).
"""

from copy import deepcopy
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.data.scientific_partitions import create_scientific_partition_indices
from analysis.scientific_metrics import (
    compute_parameter_distances,
    compute_interpolated_barriers,
    compute_angular_cka,
    evaluate_cumulative_partial_merges,
    compute_head_share
)


@pytest.fixture
def dummy_data():
    """Generates a small synthetic dataset for testing."""
    torch.manual_seed(42)
    x = torch.randn(40, 3, 32, 32)
    y = torch.randint(0, 10, (40,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=10, shuffle=False)
    return ds, loader


def test_identical_models_zero_barrier(dummy_data):
    """Identical models must have D_pair = 0, loss barrier = 0, and acc dip = 0."""
    _, loader = dummy_data
    model = ResNet20LN(num_classes=10)
    device = torch.device("cpu")

    dist = compute_parameter_distances(model, model)
    assert dist["d_pair"] == 0.0
    assert dist["d_body"] == 0.0
    assert dist["d_head"] == 0.0

    barriers = compute_interpolated_barriers(model, model, loader, device, num_points=5)
    assert abs(barriers["max_loss_barrier"]) < 1e-6
    assert abs(barriers["max_acc_dip"]) < 1e-6


def test_symmetric_pure_logit_offset_pair(dummy_data):
    """Models differing only by opposite logit offsets cancel out under symmetric averaging."""
    _, loader = dummy_data
    model_A = ResNet20LN(num_classes=10)
    model_B = deepcopy(model_A)
    device = torch.device("cpu")

    # Add opposite biases to head
    offset = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
    model_A.fc.bias.data += offset
    model_B.fc.bias.data -= offset

    # Midpoint merge cancels the offsets exactly
    barriers = compute_interpolated_barriers(model_A, model_B, loader, device, num_points=3)
    # Midpoint accuracy dip should be <= 0 (or negligible)
    assert barriers["acc_dips"][1] <= 1e-4


def test_angular_cka_identity_and_orthogonal():
    """Verifies Angular CKA on identity (1.0) and orthogonal (0.0) representations."""
    torch.manual_seed(42)
    X = torch.randn(50, 64)
    # Identity
    cka_ident = compute_angular_cka(X, X)
    assert abs(cka_ident - 1.0) < 1e-3

    # Orthogonal
    Y = torch.randn(50, 64)
    # Project Y to be orthogonal to X
    Y_orth = Y - (X @ torch.linalg.pinv(X)) @ Y
    cka_orth = compute_angular_cka(X, Y_orth)
    assert abs(cka_orth) < 0.15


def test_cumulative_partial_merges(dummy_data):
    """Cumulative partial merges on identical models match base accuracy at every stage."""
    _, loader = dummy_data
    model = ResNet20LN(num_classes=10)
    device = torch.device("cpu")

    results = evaluate_cumulative_partial_merges(model, model, loader, device)
    accs = list(results.values())
    for a in accs:
        assert abs(a - accs[0]) < 1e-6


def test_head_share_computation():
    """Verifies head share formula clipping and calculation."""
    # probe_gap = 0.05, total_gap = 0.10 -> head_share = 0.5
    hs = compute_head_share(probe_gap=0.05, no_branch_acc=0.85, merged_acc=0.75)
    assert abs(hs - 0.5) < 1e-5

    # probe_gap > total_gap clips to 1.0
    hs_clip = compute_head_share(probe_gap=0.15, no_branch_acc=0.85, merged_acc=0.75)
    assert hs_clip == 1.0


def test_scientific_partitions_invariants():
    """Verifies partition invariants: R2b balanced split, R3-L disjointness, and 70/30 split."""
    # 10 classes, 100 samples per class = 1000 samples
    targets = np.repeat(np.arange(10), 100)

    # 1. Rung 2b: exactly 500 in A, 500 in B, 50 per class
    idx_A, idx_B = create_scientific_partition_indices(targets, source="R2b", seed=42)
    assert len(idx_A) == 500
    assert len(idx_B) == 500
    assert len(set(idx_A).intersection(set(idx_B))) == 0
    t_A = targets[idx_A]
    for c in range(10):
        assert (t_A == c).sum() == 50

    # 2. Rung 3-L at s=1.0: disjoint class sets
    idx_A_s1, idx_B_s1 = create_scientific_partition_indices(targets, source="R3-L", s=1.0, seed=42)
    t_A_s1 = set(targets[idx_A_s1])
    t_B_s1 = set(targets[idx_B_s1])
    assert t_A_s1 == {0, 1, 2, 3, 4}
    assert t_B_s1 == {5, 6, 7, 8, 9}

    # 3. Asymmetric 70/30 split
    idx_A_70, idx_B_30 = create_scientific_partition_indices(targets, source="R3-L", asymmetric_mode="ratio_70_30", seed=42)
    assert len(idx_A_70) == 700
    assert len(idx_B_30) == 300
