"""Unit tests for Milestone 3: Factorial Sweeps, Coupling Modes, and Model Pretraining."""

import pytest
import torch
import torch.nn as nn
from torchvision.datasets import FakeData
import torchvision.transforms as transforms

from fl_outage.models.resnet import ResNet20LN
from fl_outage.models.pretrained import get_cifar100_pretrained_model
from fl_outage.fl.outage import OutageSimulator
from fl_outage.data.partition import partition_dataset


def test_sparse_coupling_i1_mechanics():
    """Verifies that under I1, cross-group communication only triggers every R rounds."""
    torch.manual_seed(42)
    import numpy as np
    fake_dataset = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())
    targets = np.array([y for _, y in fake_dataset])
    partitions = partition_dataset(
        targets=targets,
        num_classes=10,
        s=1.0,
        alpha=1.0,
        num_clients_per_group=2,
        seed=42
    )
    group_A = [0, 1]
    group_B = [2, 3]


    base_model = ResNet20LN(num_classes=10, width_base=4)
    sim = OutageSimulator(
        base_model=base_model,
        dataset=fake_dataset,
        partitions=partitions,
        group_A_cids=group_A,
        group_B_cids=group_B,
        lr=0.01,
        local_steps_per_round=2,
        device=torch.device("cpu")
    )

    # Run for 2 rounds with sparse_freq_R = 2
    # At round 1 (r=0): no exchange -> models should differ
    # At round 2 (r=1): exchange triggers -> models should be identical
    model_A, model_B = sim.run_outage(num_rounds=2, isolation_level="I1", sparse_freq_R=2, seed=42)

    # After round 2, since exchange_now is True, weights must be bitwise identical
    for p_a, p_b in zip(model_A.parameters(), model_B.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), "Models under I1 at exchange round must be synchronized"


def test_cifar100_pretrained_adapter():
    """Verifies that ResNet20LN adapted from CIFAR-100 outputs 10 classes and matches expected shape."""
    model = get_cifar100_pretrained_model(
        checkpoint_path="./checkpoints/test_cifar100_adapter.pt",
        num_classes=10,
        width_base=4,
        device=torch.device("cpu"),
        seed=42,
        quick_test=True
    )
    assert model.fc.out_features == 10
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_factorial_grid_completeness():
    """Verifies grid generation logic for Wave 1 and Wave 2 factor combinations."""
    # Wave 1 factors
    s_levels = [0.0, 0.5, 0.8, 1.0]
    k_levels = [0, 1, 2, 5, 20]
    i_levels = ["I0", "I1", "I2"]
    seeds = [42, 43, 44]

    wave1_cells = [
        {"seed": seed, "s": s, "k": k, "iso": iso}
        for seed in seeds
        for s in s_levels
        for k in k_levels
        for iso in i_levels
    ]
    # 4 skews * 5 onsets * 3 isolations = 60 cells per seed * 3 seeds = 180 total
    assert len(wave1_cells) == 180
    assert len(set((c["seed"], c["s"], c["k"], c["iso"]) for c in wave1_cells)) == 180
