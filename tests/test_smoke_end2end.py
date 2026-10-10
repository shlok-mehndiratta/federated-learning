import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.trunk import FedAvgTrunk
from fl_outage.fl.outage import OutageSimulator
from fl_outage.fl.merge import merge_models
from fl_outage.fl.recovery import run_recovery_evaluation
from fl_outage.metrics.barrier import compute_linear_barrier
from fl_outage.metrics.distances import compute_normalized_distances
from fl_outage.metrics.curvature import compute_curvature_barrier_estimate


def test_smoke_end_to_end_cycle():
    """Gate G0 Test 5: Complete end-to-end cycle executes cleanly and logs every metric."""
    torch.manual_seed(42)

    # Synthetic dataset with 80 samples
    x = torch.randn(80, 3, 32, 32)
    y = torch.randint(0, 10, (80,))
    train_dataset = TensorDataset(x, y)

    # Eval dataset
    x_val = torch.randn(20, 3, 32, 32)
    y_val = torch.randint(0, 10, (20,))
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=10)

    # 4 clients (Group A: 0, 1; Group B: 2, 3)
    partitions = {
        0: list(range(0, 20)),
        1: list(range(20, 40)),
        2: list(range(40, 60)),
        3: list(range(60, 80))
    }

    # 1. Initialize model & trunk
    model = ResNet20LN(num_classes=10, width_base=4)
    trunk = FedAvgTrunk(
        model=model,
        dataset=train_dataset,
        partitions=partitions,
        eval_loader=val_loader,
        lr=0.05,
        local_steps_per_round=4,
        device=torch.device("cpu"),
        checkpoint_dir="./checkpoints/test_smoke"
    )

    trunk_res = trunk.run_rounds(num_rounds=2, checkpoint_at_steps=[8], seed=42)
    assert 8 in trunk_res["checkpoints"], "Trunk failed to save checkpoint at step 8"

    # 2. Outage simulation from trunk model
    sim = OutageSimulator(
        base_model=trunk.model,
        dataset=train_dataset,
        partitions=partitions,
        group_A_cids=[0, 1],
        group_B_cids=[2, 3],
        lr=0.05,
        local_steps_per_round=4,
        device=torch.device("cpu")
    )
    model_A, model_B = sim.run_outage(num_rounds=2, isolation_level="I0", seed=100)

    # 3. Measurement: Barrier, Distances, Curvature
    barrier_dict = compute_linear_barrier(model_A, model_B, val_loader, num_points=5)
    dist_dict = compute_normalized_distances(model_A, model_B, model_anchor=trunk.model)
    curv_est = compute_curvature_barrier_estimate(model_A, model_B, val_loader, num_batches=1)

    assert not math.isnan(barrier_dict["barrier"]), "Barrier is NaN"
    assert not math.isnan(dist_dict["d_pair"]), "d_pair is NaN"
    assert not math.isnan(dist_dict["d_origin"]), "d_origin is NaN"
    assert not math.isnan(curv_est), "Curvature estimate is NaN"

    # 4. Model Reconciliation / Merge
    merged_model = merge_models(model_A, model_B, rule="plain_avg")

    # 5. Recovery Evaluation (RTB)
    recovery_res = run_recovery_evaluation(
        merged_model=merged_model,
        dataset=train_dataset,
        partitions=partitions,
        eval_loader=val_loader,
        baseline_target_acc=0.5,
        max_recovery_rounds=2,
        recovery_tolerance=0.05,
        lr=0.05,
        local_steps_per_round=4,
        device=torch.device("cpu"),
        seed=200
    )

    assert "rtb" in recovery_res
    assert "recovered" in recovery_res
    assert len(recovery_res["history"]) >= 1
    print("\nEnd-to-end smoke test passed successfully with all metrics logged!")
