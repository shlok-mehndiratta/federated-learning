import math
from typing import Dict, Optional
import torch
import torch.nn as nn


def compute_model_l2_distance(model_A: nn.Module, model_B: nn.Module) -> float:
    """Computes raw Euclidean L2 distance between parameter vectors of two models."""
    total_sq_diff = 0.0
    for p_A, p_B in zip(model_A.parameters(), model_B.parameters()):
        diff = p_A.detach() - p_B.detach()
        total_sq_diff += torch.sum(diff ** 2).item()
    return math.sqrt(total_sq_diff)


def compute_normalized_distances(
    model_A: nn.Module,
    model_B: nn.Module,
    model_anchor: Optional[nn.Module] = None
) -> Dict[str, float]:
    """Computes normalized distance metrics:
    - D_pair: ||W_A - W_B||_2 / sqrt(P)
    - D_origin: (||W_A - W_k||_2 + ||W_B - W_k||_2) / (2 * sqrt(P)) [if anchor provided]
    where P is total parameter count.
    """
    total_params = sum(p.numel() for p in model_A.parameters())
    sqrt_P = math.sqrt(total_params)

    raw_dist = compute_model_l2_distance(model_A, model_B)
    d_pair = raw_dist / sqrt_P

    result = {
        "num_params": total_params,
        "raw_l2_distance": raw_dist,
        "d_pair": d_pair,
    }

    if model_anchor is not None:
        dist_A_origin = compute_model_l2_distance(model_A, model_anchor)
        dist_B_origin = compute_model_l2_distance(model_B, model_anchor)
        d_origin = (dist_A_origin + dist_B_origin) / (2.0 * sqrt_P)
        result["dist_A_origin"] = dist_A_origin
        result["dist_B_origin"] = dist_B_origin
        result["d_origin"] = d_origin

    return result
