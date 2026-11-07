"""Unit tests for Milestone 4 mechanisms, representational analysis (CKA), and recovery acceleration."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.metrics.cka import linear_cka, extract_stage_features, compute_stage_cka_summary
from fl_outage.metrics.layer_barrier import (
    get_param_stage,
    interpolate_single_stage,
    compute_layer_wise_barrier_profile
)
from fl_outage.fl.rebasin import match_weights_resnet20, apply_permutations_to_resnet20
from fl_outage.fl.client import FederatedClient


def test_linear_cka_identity():
    """Verifies that CKA(X, X) == 1.0 exactly."""
    torch.manual_seed(42)
    X = torch.randn(100, 32)
    score = linear_cka(X, X)
    assert pytest.approx(1.0, abs=1e-5) == score


def test_linear_cka_orthogonal_invariance():
    """Verifies that linear CKA is invariant to orthogonal coordinate rotations."""
    torch.manual_seed(42)
    X = torch.randn(100, 16)
    # Generate random orthogonal matrix Q via QR decomposition
    A = torch.randn(16, 16)
    Q, _ = torch.linalg.qr(A)
    Y = torch.matmul(X, Q)

    score = linear_cka(X, Y)
    assert pytest.approx(1.0, abs=1e-4) == score


def test_linear_cka_uncorrelated():
    """Verifies that independent random features have low CKA."""
    torch.manual_seed(42)
    X = torch.randn(500, 32)
    Y = torch.randn(500, 32)
    score = linear_cka(X, Y)
    assert score < 0.25


def test_param_stage_mapping():
    """Verifies parameter name mapping to ResNet20LN stage blocks."""
    assert get_param_stage("init_conv.weight") == "init"
    assert get_param_stage("init_norm.weight") == "init"
    assert get_param_stage("stages.0.conv1.weight") == "stage1"
    assert get_param_stage("stages.2.conv2.weight") == "stage1"
    assert get_param_stage("stages.3.conv1.weight") == "stage2"
    assert get_param_stage("stages.5.norm2.bias") == "stage2"
    assert get_param_stage("stages.6.shortcut.conv.weight") == "stage3"
    assert get_param_stage("stages.8.conv2.weight") == "stage3"
    assert get_param_stage("fc.weight") == "fc"
    assert get_param_stage("fc.bias") == "fc"


def test_interpolate_single_stage():
    """Verifies that single stage interpolation modifies only the targeted stage."""
    torch.manual_seed(42)
    mA = ResNet20LN(num_classes=10)
    mB = ResNet20LN(num_classes=10)

    # Interpolate stage2 with alpha=1.0 and background='endpoint_A'
    interp_state = interpolate_single_stage(mA, mB, target_stage="stage2", alpha=1.0, background_mode="endpoint_A")

    mA_state = mA.state_dict()
    mB_state = mB.state_dict()

    for k, v in interp_state.items():
        stage = get_param_stage(k)
        if stage == "stage2":
            # Must equal mB
            assert torch.allclose(v, mB_state[k], atol=1e-6)
        else:
            # Must equal mA
            assert torch.allclose(v, mA_state[k], atol=1e-6)


def test_rebasin_identity_alignment():
    """Verifies that Re-Basin on identical models preserves weights and produces zero deviation."""
    torch.manual_seed(42)
    mA = ResNet20LN(num_classes=10)
    mB = ResNet20LN(num_classes=10)
    mB.load_state_dict(mA.state_dict())

    perm_B, rebasin_merged, info = match_weights_resnet20(mA, mB, max_iters=1)
    
    # Merged model should equal mA exactly
    for k in mA.state_dict().keys():
        diff = torch.norm(rebasin_merged.state_dict()[k] - mA.state_dict()[k]).item()
        assert diff < 1e-5


def test_rebasin_parameter_shapes():
    """Verifies that permuted model preserves all parameter tensor dimensions."""
    torch.manual_seed(42)
    mA = ResNet20LN(num_classes=10)
    mB = ResNet20LN(num_classes=10)

    perm_B, rebasin_merged, info = match_weights_resnet20(mA, mB, max_iters=2)

    sd_orig = mA.state_dict()
    sd_rebasin = rebasin_merged.state_dict()

    for k in sd_orig.keys():
        assert sd_rebasin[k].shape == sd_orig[k].shape


def test_client_momentum_reset():
    """Verifies that FederatedClient accumulates momentum and resets cleanly."""
    torch.manual_seed(42)
    fake_data = torch.randn(20, 3, 32, 32)
    fake_targets = torch.randint(0, 10, (20,))
    dataset = TensorDataset(fake_data, fake_targets)

    client = FederatedClient(
        client_id=0,
        dataset=dataset,
        indices=list(range(20)),
        lr=0.01,
        momentum=0.9
    )

    model = ResNet20LN(num_classes=10)
    assert client.optimizer_state is None

    client.train_steps(model, num_steps=2, seed=42)
    assert client.optimizer_state is not None
    assert "state" in client.optimizer_state

    # Reset momentum
    client.reset_momentum()
    assert client.optimizer_state is None
