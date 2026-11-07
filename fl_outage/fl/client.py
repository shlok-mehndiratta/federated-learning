from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_outage.data.cifar import create_deterministic_loader


class FederatedClient:
    """Represents a single federated client performing local SGD."""

    def __init__(
        self,
        client_id: int,
        dataset: torch.utils.data.Dataset,
        indices: List[int],
        batch_size: int = 128,
        lr: float = 0.1,
        momentum: float = 0.0,
        device: torch.device = torch.device("cpu")
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.subset = Subset(dataset, indices)
        self.optimizer_state: Optional[Dict] = None

    def reset_momentum(self):
        """Clears local optimizer momentum buffers upon server reconnection."""
        self.optimizer_state = None

    def train_steps(
        self,
        model: nn.Module,
        num_steps: int,
        seed: int = 42
    ) -> Dict[str, float]:
        """Trains model sequentially for num_steps local gradient steps.
        
        Controls:
        - Plain SGD without momentum.
        - Constant learning rate lr.
        - Returns training statistics (loss, steps).
        """
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0)
        if self.optimizer_state is not None and self.momentum > 0.0:
            try:
                optimizer.load_state_dict(self.optimizer_state)
            except Exception:
                pass
        criterion = nn.CrossEntropyLoss()

        loader = create_deterministic_loader(
            self.subset,
            batch_size=min(self.batch_size, len(self.indices)),
            shuffle=True,
            seed=seed,
            drop_last=False
        )

        step_count = 0
        total_loss = 0.0
        data_iter = iter(loader)

        while step_count < num_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

        if self.momentum > 0.0:
            self.optimizer_state = optimizer.state_dict()

        return {
            "client_id": self.client_id,
            "steps": step_count,
            "avg_loss": total_loss / max(step_count, 1)
        }
