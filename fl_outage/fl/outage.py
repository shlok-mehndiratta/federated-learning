from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from fl_outage.fl.client import FederatedClient
from fl_outage.fl.trunk import average_model_weights


class OutageSimulator:
    """Simulates a parameter server outage where clients split into Group A and Group B."""

    def __init__(
        self,
        base_model: nn.Module,
        dataset: Dataset,
        partitions: Dict[int, List[int]],
        group_A_cids: List[int],
        group_B_cids: List[int],
        lr: float = 0.1,
        local_steps_per_round: int = 10,
        device: torch.device = torch.device("cpu")
    ):
        self.device = device
        self.lr = lr
        self.local_steps = local_steps_per_round

        # Model replicas for Group A and Group B
        self.model_A = deepcopy(base_model).to(device)
        self.model_B = deepcopy(base_model).to(device)

        self.clients_A = [
            FederatedClient(cid, dataset, partitions[cid], lr=lr, device=device)
            for cid in sorted(group_A_cids)
        ]
        self.clients_B = [
            FederatedClient(cid, dataset, partitions[cid], lr=lr, device=device)
            for cid in sorted(group_B_cids)
        ]

    def run_outage(
        self,
        num_rounds: int,
        isolation_level: str = "I0",
        sparse_freq_R: int = 5,
        seed: int = 42
    ) -> Tuple[nn.Module, nn.Module]:
        """Runs outage simulation for num_rounds.
        
        Isolation levels:
        - 'I0': Fully isolated (no cross-group communication).
        - 'I1': Sparse cross-group exchange every R rounds.
        - 'I2': Dense cross-group exchange every round (Randl coupled baseline).
        """
        assert isolation_level in ("I0", "I1", "I2"), f"Invalid isolation level: {isolation_level}"

        for r in range(num_rounds):
            round_seed = seed + r * 1000

            # 1. Group A internal training & aggregation
            states_A: List[OrderedDict] = []
            base_A = deepcopy(self.model_A.state_dict())
            for client in self.clients_A:
                self.model_A.load_state_dict(base_A)
                client.train_steps(self.model_A, num_steps=self.local_steps, seed=round_seed + client.client_id)
                states_A.append(deepcopy(self.model_A.state_dict()))
            avg_A = average_model_weights(states_A)
            self.model_A.load_state_dict(avg_A)

            # 2. Group B internal training & aggregation
            states_B: List[OrderedDict] = []
            base_B = deepcopy(self.model_B.state_dict())
            for client in self.clients_B:
                self.model_B.load_state_dict(base_B)
                client.train_steps(self.model_B, num_steps=self.local_steps, seed=round_seed + client.client_id)
                states_B.append(deepcopy(self.model_B.state_dict()))
            avg_B = average_model_weights(states_B)
            self.model_B.load_state_dict(avg_B)

            # 3. Cross-group communication according to isolation level
            exchange_now = False
            if isolation_level == "I2":
                exchange_now = True
            elif isolation_level == "I1":
                if (r + 1) % sparse_freq_R == 0:
                    exchange_now = True

            if exchange_now:
                # Cross-group parameter averaging
                cross_avg = average_model_weights([self.model_A.state_dict(), self.model_B.state_dict()])
                self.model_A.load_state_dict(cross_avg)
                self.model_B.load_state_dict(cross_avg)

        return self.model_A, self.model_B
