"""Layer-wise and stage-wise barrier decomposition for ResNet20LN.

Decomposes the loss and accuracy barriers along the linear interpolation path
into architectural blocks:
- 'init': Initial Conv (3->16) and LayerNorm
- 'stage1': Stage 1 residual blocks (16 filters)
- 'stage2': Stage 2 residual blocks (32 filters)
- 'stage3': Stage 3 residual blocks (64 filters)
- 'fc': Final Linear classification head (64->num_classes)
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_outage.metrics.barrier import evaluate_loss_and_acc


STAGE_BLOCK_PREFIXES = {
    "init": ("init_conv", "init_norm"),
    "stage1": ("stages.0.", "stages.1.", "stages.2."),
    "stage2": ("stages.3.", "stages.4.", "stages.5."),
    "stage3": ("stages.6.", "stages.7.", "stages.8."),
    "fc": ("fc.",)
}


def get_param_stage(param_name: str) -> str:
    """Maps parameter name to ResNet20LN stage block."""
    for stage, prefixes in STAGE_BLOCK_PREFIXES.items():
        if any(param_name.startswith(p) for p in prefixes):
            return stage
    return "other"


@torch.no_grad()
def interpolate_single_stage(
    model_A: nn.Module,
    model_B: nn.Module,
    target_stage: str,
    alpha: float,
    background_mode: str = "midpoint"
) -> OrderedDict:
    """Interpolates weights for a single stage block while holding others fixed.
    
    Args:
        model_A: Endpoint Model A.
        model_B: Endpoint Model B.
        target_stage: Stage to interpolate ('init', 'stage1', 'stage2', 'stage3', 'fc').
        alpha: Interpolation coefficient in [0.0, 1.0] for target_stage.
        background_mode: How other stages are set:
            - 'midpoint': Fixed at 0.5 * (W_A + W_B)
            - 'endpoint_A': Fixed at W_A
            - 'endpoint_B': Fixed at W_B
    """
    state_A = model_A.state_dict()
    state_B = model_B.state_dict()
    interp_state = OrderedDict()

    for k in state_A.keys():
        val_A = state_A[k]
        val_B = state_B[k]
        param_stage = get_param_stage(k)

        if not torch.is_floating_point(val_A):
            interp_state[k] = val_A
            continue

        if param_stage == target_stage:
            interp_state[k] = (1.0 - alpha) * val_A + alpha * val_B
        else:
            if background_mode == "midpoint":
                interp_state[k] = 0.5 * (val_A + val_B)
            elif background_mode == "endpoint_A":
                interp_state[k] = val_A
            elif background_mode == "endpoint_B":
                interp_state[k] = val_B
            else:
                raise ValueError(f"Unknown background mode: {background_mode}")

    return interp_state


def compute_layer_wise_barrier_profile(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    num_points: int = 11,
    device: torch.device = torch.device("cpu")
) -> Dict[str, any]:
    """Evaluates the barrier contribution of each architectural stage.
    
    Returns:
        Dictionary containing:
        - 'stage_profiles': per-stage interpolation curves (alpha, loss, acc)
        - 'stage_acc_dips': maximum accuracy dip per stage when interpolated against midpoint
        - 'isolated_merge_impact': impact of merging ONLY that stage while keeping rest at W_A
    """
    stages = ["init", "stage1", "stage2", "stage3", "fc"]
    alphas = [i / (num_points - 1) for i in range(num_points)]
    eval_model = deepcopy(model_A).to(device)

    # Baseline endpoint and midpoint metrics
    _, acc_A = evaluate_loss_and_acc(model_A, dataloader, device=device)
    _, acc_B = evaluate_loss_and_acc(model_B, dataloader, device=device)

    # 1. Evaluate per-stage interpolation against midpoint background
    stage_profiles = {}
    stage_acc_dips = {}

    for stage in stages:
        losses = []
        accs = []
        for alpha in alphas:
            state = interpolate_single_stage(model_A, model_B, target_stage=stage, alpha=alpha, background_mode="midpoint")
            eval_model.load_state_dict(state)
            l, a = evaluate_loss_and_acc(eval_model, dataloader, device=device)
            losses.append(l)
            accs.append(a)

        # Accuracy dip along the stage interpolation path
        acc_0 = accs[0]
        acc_1 = accs[-1]
        dips = [((1.0 - alpha) * acc_0 + alpha * acc_1) - a for alpha, a in zip(alphas, accs)]
        max_dip = max(dips)

        stage_profiles[stage] = {
            "alphas": alphas,
            "losses": losses,
            "accs": accs,
            "midpoint_acc": accs[num_points // 2],
            "max_dip": max_dip
        }
        stage_acc_dips[stage] = max_dip

    # 2. Evaluate isolated merge impact (merging ONLY stage into W_A)
    isolated_merge_impact = {}
    for stage in stages:
        # alpha=0.5 for target_stage, background='endpoint_A'
        state = interpolate_single_stage(model_A, model_B, target_stage=stage, alpha=0.5, background_mode="endpoint_A")
        eval_model.load_state_dict(state)
        _, merged_stage_acc = evaluate_loss_and_acc(eval_model, dataloader, device=device)
        isolated_merge_impact[stage] = {
            "acc_with_stage_merged": merged_stage_acc,
            "drop_from_A": acc_A - merged_stage_acc
        }

    return {
        "stages": stages,
        "acc_A": acc_A,
        "acc_B": acc_B,
        "stage_acc_dips": stage_acc_dips,
        "stage_profiles": stage_profiles,
        "isolated_merge_impact": isolated_merge_impact
    }
