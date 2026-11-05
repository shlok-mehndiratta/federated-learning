import pytest
import torch
from torch.utils.data import TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.server import FederatedServer


def test_server_training_determinism():
    """Gate G0 Test 1: Two identical runs with the same seed yield bitwise identical model weights."""
    torch.manual_seed(42)

    # Synthetic toy dataset of 50 samples
    x = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 10, (50,))
    toy_dataset = TensorDataset(x, y)

    partitions = {
        0: list(range(0, 25)),
        1: list(range(25, 50))
    }

    # Run 1
    torch.manual_seed(100)
    model1 = ResNet20LN(num_classes=10, width_base=4)  # Small width for quick test
    server1 = FederatedServer(
        model=model1,
        dataset=toy_dataset,
        partitions=partitions,
        local_steps_per_round=5,
        device=torch.device("cpu")
    )
    server1.run_rounds(num_rounds=2, seed=777)
    weights1 = {k: v.clone() for k, v in server1.model.state_dict().items()}

    # Run 2 with identical seeds
    torch.manual_seed(100)
    model2 = ResNet20LN(num_classes=10, width_base=4)
    server2 = FederatedServer(
        model=model2,
        dataset=toy_dataset,
        partitions=partitions,
        local_steps_per_round=5,
        device=torch.device("cpu")
    )
    server2.run_rounds(num_rounds=2, seed=777)
    weights2 = server2.model.state_dict()

    # Verify bitwise identity across all parameters
    for k in weights1.keys():
        diff = torch.max(torch.abs(weights1[k] - weights2[k])).item()
        assert diff == 0.0, f"Parameter {k} differed between deterministic runs: max diff = {diff}"
