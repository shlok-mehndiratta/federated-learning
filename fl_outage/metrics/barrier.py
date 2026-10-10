from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def interpolate_weights(model_A: nn.Module, model_B: nn.Module, alpha: float) -> OrderedDict:
    """Linearly interpolates state dicts: W(alpha) = (1 - alpha) * W_A + alpha * W_B."""
    state_dict = OrderedDict()
    params_A = model_A.state_dict()
    params_B = model_B.state_dict()

    for k in params_A.keys():
        val_A = params_A[k]
        val_B = params_B[k]
        if torch.is_floating_point(val_A):
            state_dict[k] = (1.0 - alpha) * val_A + alpha * val_B
        else:
            # For integer buffers
            state_dict[k] = val_A
    return state_dict


@torch.no_grad()
def evaluate_loss_and_acc(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float]:
    """Computes average cross-entropy loss and top-1 accuracy over dataloader."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def compute_linear_barrier(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    num_points: int = 11,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """Computes Sharma et al. endpoint-interpolated loss barrier at equidistant points.
    
    Formula:
        Barrier = max_{alpha in [0, 1]} ( L(W(alpha)) - [(1 - alpha)*L(W_A) + alpha*L(W_B)] )
    """
    alphas = [i / (num_points - 1) for i in range(num_points)]
    losses: List[float] = []
    accs: List[float] = []

    eval_model = deepcopy(model_A).to(device)
    criterion = nn.CrossEntropyLoss()

    for alpha in alphas:
        interp_state = interpolate_weights(model_A, model_B, alpha)
        eval_model.load_state_dict(interp_state)
        l, acc = evaluate_loss_and_acc(eval_model, dataloader, device=device, criterion=criterion)
        losses.append(l)
        accs.append(acc)

    loss_0 = losses[0]
    loss_1 = losses[-1]

    barriers = [
        loss - ((1.0 - alpha) * loss_0 + alpha * loss_1)
        for alpha, loss in zip(alphas, losses)
    ]
    max_barrier = max(barriers)
    max_idx = barriers.index(max_barrier)

    # Error / Accuracy barrier (dip below linear path)
    acc_0 = accs[0]
    acc_1 = accs[-1]
    acc_dips = [
        ((1.0 - alpha) * acc_0 + alpha * acc_1) - acc
        for alpha, acc in zip(alphas, accs)
    ]
    max_acc_dip = max(acc_dips)

    return {
        "barrier": float(max_barrier),
        "barrier_alpha": float(alphas[max_idx]),
        "max_loss": float(max(losses)),
        "loss_endpoint_0": float(loss_0),
        "loss_endpoint_1": float(loss_1),
        "merged_loss": float(losses[num_points // 2]),
        "merged_acc": float(accs[num_points // 2]),
        "max_acc_dip": float(max_acc_dip),
        "all_losses": losses,
        "all_accs": accs,
    }
