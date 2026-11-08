"""
Unit tests for Stage 2 Head Cures runner and decomposition pipeline.
Verifies:
1. create_few_shot_calibration_loader builds exactly 20 samples per class (200 total).
2. evaluate_stage2_mechanisms completes end-to-end without shape errors on ResNet20LN.
"""

from copy import deepcopy
import importlib
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_outage.models.resnet import ResNet20LN

stage2 = importlib.import_module("experiments.02_stage2_head_cures")
create_few_shot_calibration_loader = stage2.create_few_shot_calibration_loader
build_probe_set = stage2.build_probe_set
evaluate_stage2_mechanisms = stage2.evaluate_stage2_mechanisms


class DummyDataset(TensorDataset):
    def __init__(self, n_samples=300):
        torch.manual_seed(42)
        x = torch.randn(n_samples, 3, 32, 32)
        y = torch.tensor([i % 10 for i in range(n_samples)], dtype=torch.long)
        super().__init__(x, y)
        self.targets = y.numpy()


def test_few_shot_calibration_loader_balance():
    ds = DummyDataset(n_samples=300)
    loader = create_few_shot_calibration_loader(ds, samples_per_class=20, num_classes=10, seed=42)
    total = 0
    class_counts = torch.zeros(10)
    for _, y in loader:
        total += y.size(0)
        for c in range(10):
            class_counts[c] += (y == c).sum()

    assert total == 200
    assert torch.all(class_counts == 20)


def test_stage2_mechanisms_end2end():
    ds = DummyDataset(n_samples=60)
    loader = DataLoader(ds, batch_size=20)
    probe_loader = DataLoader(ds, batch_size=20)
    few_shot_loader = DataLoader(ds, batch_size=20)

    model_A = ResNet20LN(num_classes=10)
    model_B = deepcopy(model_A)
    device = torch.device("cpu")

    results = evaluate_stage2_mechanisms(
        model_A=model_A,
        model_B=model_B,
        test_loader=loader,
        probe_loader=probe_loader,
        few_shot_loader=few_shot_loader,
        train_balanced_loader=loader,
        source="R3-L",
        s=0.8,
        asymmetric_mode=None,
        device=device
    )

    assert "endpoints" in results
    assert "cure1_zero_shot_logit_adj" in results
    assert "cure2_few_shot_recalibration" in results
    assert "cure3_decoupled_head_ensemble" in results
    assert "cure4_linear_probe_upper_bound" in results
    assert "layerwise_cka" in results
    assert len(results["layerwise_cka"]) == 6
