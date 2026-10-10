from collections import OrderedDict
from copy import deepcopy
import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from fl_outage.fl.client import FederatedClient
from fl_outage.metrics.barrier import evaluate_loss_and_acc


def average_model_weights(client_models_state: List[OrderedDict]) -> OrderedDict:
    """Averages an array of model state_dicts (FedAvg)."""
    assert len(client_models_state) > 0, "No client states to average"
    avg_state = OrderedDict()
    num_clients = len(client_models_state)

    for key in client_models_state[0].keys():
        first_tensor = client_models_state[0][key]
        if torch.is_floating_point(first_tensor):
            stacked = torch.stack([state[key] for state in client_models_state], dim=0)
            avg_state[key] = torch.mean(stacked, dim=0)
        else:
            avg_state[key] = first_tensor
    return avg_state


class FedAvgTrunk:
    """Manages the global federated learning trunk prior to any server outage."""

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        partitions: Dict[int, List[int]],
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
        lr: float = 0.1,
        local_steps_per_round: int = 10,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.partitions = partitions
        self.eval_loader = eval_loader
        self.lr = lr
        self.local_steps = local_steps_per_round
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.clients = [
            FederatedClient(
                client_id=cid,
                dataset=dataset,
                indices=indices,
                lr=lr,
                device=device
            )
            for cid, indices in sorted(partitions.items())
        ]

    def save_checkpoint(self, step: int, tag: str = "") -> str:
        """Saves current global trunk model state."""
        fname = f"trunk_step_{step}_{tag}.pt" if tag else f"trunk_step_{step}.pt"
        path = os.path.join(self.checkpoint_dir, fname)
        torch.save(self.model.state_dict(), path)
        return path

    def run_rounds(
        self,
        num_rounds: int,
        checkpoint_at_steps: Optional[List[int]] = None,
        start_step: int = 0,
        seed: int = 42
    ) -> Dict[str, any]:
        """Runs FedAvg for num_rounds, checkpointing at specified step milestones."""
        if checkpoint_at_steps is None:
            checkpoint_at_steps = []

        checkpoints_saved = {}
        total_steps = start_step

        # Checkpoint at step 0 if requested
        if total_steps in checkpoint_at_steps:
            path = self.save_checkpoint(total_steps)
            checkpoints_saved[total_steps] = path

        round_history = []

        for r in range(num_rounds):
            round_seed = seed + r * 1000
            client_states: List[OrderedDict] = []

            # Sequential client training against shared model state
            base_state = deepcopy(self.model.state_dict())

            for client in self.clients:
                # Load shared base weights
                self.model.load_state_dict(base_state)
                # Train client locally
                client.train_steps(
                    self.model,
                    num_steps=self.local_steps,
                    seed=round_seed + client.client_id
                )
                # Save client state
                client_states.append(deepcopy(self.model.state_dict()))

            # Server-side FedAvg
            avg_state = average_model_weights(client_states)
            self.model.load_state_dict(avg_state)

            total_steps += self.local_steps

            # Checkpoint milestone check
            if total_steps in checkpoint_at_steps:
                path = self.save_checkpoint(total_steps)
                checkpoints_saved[total_steps] = path

            # Evaluate if loader provided
            eval_metrics = {}
            if self.eval_loader is not None:
                loss, acc = evaluate_loss_and_acc(self.model, self.eval_loader, device=self.device)
                eval_metrics = {"loss": loss, "acc": acc}

            round_history.append({
                "round": r + 1,
                "step": total_steps,
                **eval_metrics
            })

        return {
            "final_step": total_steps,
            "checkpoints": checkpoints_saved,
            "history": round_history
        }
