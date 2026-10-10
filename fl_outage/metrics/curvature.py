from copy import deepcopy
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_outage.metrics.barrier import interpolate_weights


def compute_curvature_barrier_estimate(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    criterion: nn.Module = None,
    num_batches: int = 2
) -> float:
    """Computes Singh et al. (ICML 2024 HiLD) curvature-weighted barrier estimate.
    
    Uses 2nd-order Taylor expansion around the midpoint W_mid = 0.5 * (W_A + W_B):
        Barrier_est ≈ (1 / 8) * v^T ∇^2 L(W_mid) v
    where v = W_B - W_A, evaluated using a single Hessian-Vector Product (HVP).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Direction vector v = W_B - W_A
    v_tensors: List[torch.Tensor] = []
    params_A = list(model_A.parameters())
    params_B = list(model_B.parameters())

    for p_A, p_B in zip(params_A, params_B):
        diff = (p_B.detach() - p_A.detach()).to(device)
        v_tensors.append(diff)

    # Construct midpoint model W_mid = 0.5 * (W_A + W_B)
    mid_model = deepcopy(model_A).to(device)
    mid_state = interpolate_weights(model_A, model_B, 0.5)
    mid_model.load_state_dict(mid_state)
    mid_model.train()  # require grad

    mid_params = [p for p in mid_model.parameters() if p.requires_grad]
    active_v = [v for p, v in zip(mid_model.parameters(), v_tensors) if p.requires_grad]

    total_vHv = 0.0
    batches_evaluated = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        mid_model.zero_grad()
        out = mid_model(x)
        loss = criterion(out, y)

        # First gradient: g = ∇ L(W_mid)
        grads = torch.autograd.grad(loss, mid_params, create_graph=True, retain_graph=True)

        # Dot product with direction: g^T v
        gv = torch.tensor(0.0, device=device)
        for g, v in zip(grads, active_v):
            gv = gv + torch.sum(g * v)

        # Second gradient (HVP): H v = ∇ (g^T v)
        hvp = torch.autograd.grad(gv, mid_params, retain_graph=False)

        # Quadratic form: v^T H v
        batch_vHv = 0.0
        for h, v in zip(hvp, active_v):
            batch_vHv += torch.sum(h.detach() * v).item()

        total_vHv += batch_vHv
        batches_evaluated += 1

    avg_vHv = total_vHv / max(batches_evaluated, 1)
    barrier_estimate = (1.0 / 8.0) * avg_vHv
    return float(barrier_estimate)
