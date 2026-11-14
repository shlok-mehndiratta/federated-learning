from copy import deepcopy
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.metrics.barrier import compute_linear_barrier
from fl_outage.metrics.curvature import compute_curvature_barrier_estimate


# verify barrier computation properties between identical and perturbed models
def test_barrier_self_is_zero():
    """Barrier between a model and itself is identically zero."""
    torch.manual_seed(42)
    model = ResNet20LN(num_classes=10, width_base=4)

    # Toy evaluation dataloader
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    # Compute barrier against itself
    barrier_results = compute_linear_barrier(model, model, loader, num_points=5)

    assert abs(barrier_results["barrier"]) < 1e-6, f"Self-barrier should be 0.0, got {barrier_results['barrier']}"
    assert barrier_results["merged_loss"] == pytest.approx(barrier_results["loss_endpoint_0"], abs=1e-5)


def test_curvature_barrier_self_is_zero():
    """Gate G0 Test 2b: Curvature estimate for identical models is 0.0 (since v = 0)."""
    torch.manual_seed(42)
    model = ResNet20LN(num_classes=10, width_base=4)

    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    curv_est = compute_curvature_barrier_estimate(model, model, loader)
    assert abs(curv_est) < 1e-7, f"Curvature estimate for identical models must be 0, got {curv_est}"
