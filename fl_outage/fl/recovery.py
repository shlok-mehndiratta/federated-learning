from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fl_outage.fl.trunk import FedAvgTrunk
from fl_outage.metrics.barrier import evaluate_loss_and_acc


def run_recovery_evaluation(
    merged_model: nn.Module,
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    eval_loader: DataLoader,
    baseline_target_acc: float,
    max_recovery_rounds: int = 20,
    recovery_tolerance: float = 0.01,
    lr: float = 0.1,
    local_steps_per_round: int = 10,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> Dict[str, any]:
    """Runs post-merge FedAvg healing rounds until accuracy recovers to within recovery_tolerance.
    
    Returns:
    - recovered: bool
    - rtb (Rounds-to-Baseline): int (or max_recovery_rounds if not reached)
    - recovery_history: list of round evaluations
    """
    trunk = FedAvgTrunk(
        model=merged_model,
        dataset=dataset,
        partitions=partitions,
        eval_loader=eval_loader,
        lr=lr,
        local_steps_per_round=local_steps_per_round,
        device=device,
        checkpoint_dir="./checkpoints/recovery"
    )

    # Initial accuracy immediately post-merge
    init_loss, init_acc = evaluate_loss_and_acc(merged_model, eval_loader, device=device)
    target_threshold = baseline_target_acc - recovery_tolerance

    history = [{
        "round": 0,
        "loss": init_loss,
        "acc": init_acc
    }]

    if init_acc >= target_threshold:
        return {
            "recovered": True,
            "rtb": 0,
            "final_acc": init_acc,
            "history": history
        }

    recovered = False
    rtb = max_recovery_rounds

    for r in range(1, max_recovery_rounds + 1):
        round_res = trunk.run_rounds(num_rounds=1, start_step=r * local_steps_per_round, seed=seed + r * 50)
        curr_loss, curr_acc = evaluate_loss_and_acc(trunk.model, eval_loader, device=device)
        history.append({
            "round": r,
            "loss": curr_loss,
            "acc": curr_acc
        })

        if curr_acc >= target_threshold and not recovered:
            recovered = True
            rtb = r
            break

    return {
        "recovered": recovered,
        "rtb": rtb,
        "final_acc": history[-1]["acc"],
        "history": history
    }
