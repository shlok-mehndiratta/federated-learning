import pytest
import torch
from torch.utils.data import TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.outage import OutageSimulator


def test_isolation_group_A_invariant_to_group_B():
    """Gate G0 Test 4: Under isolation I0, Group A's parameters do not depend on Group B's data."""
    torch.manual_seed(42)

    # Base dataset of 60 samples
    x = torch.randn(60, 3, 32, 32)
    y1 = torch.randint(0, 10, (60,))
    dataset1 = TensorDataset(x, y1)

    # Modified dataset where Group B's data is completely changed
    y2 = y1.clone()
    y2[30:] = (y2[30:] + 1) % 10  # Perturb labels for indices 30..59 (Group B)
    dataset2 = TensorDataset(x, y2)

    partitions = {
        0: list(range(0, 15)),   # Group A
        1: list(range(15, 30)),  # Group A
        2: list(range(30, 45)),  # Group B
        3: list(range(45, 60))   # Group B
    }

    base_model = ResNet20LN(num_classes=10, width_base=4)

    # Run 1 under I0
    sim1 = OutageSimulator(
        base_model=base_model,
        dataset=dataset1,
        partitions=partitions,
        group_A_cids=[0, 1],
        group_B_cids=[2, 3],
        local_steps_per_round=5,
        device=torch.device("cpu")
    )
    model_A1, model_B1 = sim1.run_outage(num_rounds=2, isolation_level="I0", seed=42)
    weights_A1 = {k: v.clone() for k, v in model_A1.state_dict().items()}

    # Run 2 with perturbed Group B data under I0
    sim2 = OutageSimulator(
        base_model=base_model,
        dataset=dataset2,
        partitions=partitions,
        group_A_cids=[0, 1],
        group_B_cids=[2, 3],
        local_steps_per_round=5,
        device=torch.device("cpu")
    )
    model_A2, model_B2 = sim2.run_outage(num_rounds=2, isolation_level="I0", seed=42)
    weights_A2 = model_A2.state_dict()

    # Verify Group A was completely untouched by Group B's perturbation
    for k in weights_A1.keys():
        diff = torch.max(torch.abs(weights_A1[k] - weights_A2[k])).item()
        assert diff == 0.0, f"Group A parameter {k} changed when Group B was perturbed under I0! Diff = {diff}"

    # Meanwhile, verify that Group B DID change (proving the test is sensitive)
    diff_B = 0.0
    for p1, p2 in zip(model_B1.parameters(), model_B2.parameters()):
        diff_B += torch.sum(torch.abs(p1 - p2)).item()
    assert diff_B > 0.0, "Group B should have differed due to data perturbation!"
