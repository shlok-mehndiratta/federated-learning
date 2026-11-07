"""Git Re-Basin / Weight Matching for ResNet20LN.

Implements channel permutation alignment (Ainsworth et al., ICLR 2023) using the
Hungarian algorithm (linear sum assignment) to overcome neuron permutation misalignment
in model merging under federated class and client skew.
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn


def solve_permutation(cost_matrix: np.ndarray) -> np.ndarray:
    """Solves the maximum-weight bipartite matching problem using Hungarian algorithm.
    
    Args:
        cost_matrix: (C, C) similarity/correlation matrix between model A and model B channels.
        
    Returns:
        Array p of shape (C,) such that p[i] is the channel in B mapped to position i in A.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    p = np.zeros(cost_matrix.shape[0], dtype=np.int64)
    p[row_ind] = col_ind
    return p


@torch.no_grad()
def apply_permutations_to_resnet20(
    model: nn.Module,
    p_stage1: np.ndarray,
    p_stage2: np.ndarray,
    p_stage3: np.ndarray,
    p_blocks_int: Dict[int, np.ndarray]
) -> nn.Module:
    """Permutes channels and weights of ResNet20LN according to specified permutations."""
    permuted = deepcopy(model)
    sd = permuted.state_dict()

    p_s1 = torch.from_numpy(p_stage1)
    p_s2 = torch.from_numpy(p_stage2)
    p_s3 = torch.from_numpy(p_stage3)

    # 1. Initial Conv & Norm (output -> p_stage1)
    sd["init_conv.weight"] = sd["init_conv.weight"][p_s1, :, :, :]
    sd["init_norm.weight"] = sd["init_norm.weight"][p_s1]
    sd["init_norm.bias"] = sd["init_norm.bias"][p_s1]

    # 2. Stage 1 blocks (b = 0, 1, 2)
    for b in range(3):
        p_int = torch.from_numpy(p_blocks_int[b])
        prefix = f"stages.{b}."
        # conv1: in=p_s1, out=p_int
        sd[f"{prefix}conv1.weight"] = sd[f"{prefix}conv1.weight"][p_int, :, :, :][:, p_s1, :, :]
        sd[f"{prefix}norm1.weight"] = sd[f"{prefix}norm1.weight"][p_int]
        sd[f"{prefix}norm1.bias"] = sd[f"{prefix}norm1.bias"][p_int]
        # conv2: in=p_int, out=p_s1
        sd[f"{prefix}conv2.weight"] = sd[f"{prefix}conv2.weight"][p_s1, :, :, :][:, p_int, :, :]
        sd[f"{prefix}norm2.weight"] = sd[f"{prefix}norm2.weight"][p_s1]
        sd[f"{prefix}norm2.bias"] = sd[f"{prefix}norm2.bias"][p_s1]

    # 3. Block 3 (downsampling Stage 1 -> Stage 2)
    p_int3 = torch.from_numpy(p_blocks_int[3])
    sd["stages.3.conv1.weight"] = sd["stages.3.conv1.weight"][p_int3, :, :, :][:, p_s1, :, :]
    sd["stages.3.norm1.weight"] = sd["stages.3.norm1.weight"][p_int3]
    sd["stages.3.norm1.bias"] = sd["stages.3.norm1.bias"][p_int3]
    sd["stages.3.conv2.weight"] = sd["stages.3.conv2.weight"][p_s2, :, :, :][:, p_int3, :, :]
    sd["stages.3.norm2.weight"] = sd["stages.3.norm2.weight"][p_s2]
    sd["stages.3.norm2.bias"] = sd["stages.3.norm2.bias"][p_s2]
    sd["stages.3.shortcut.conv.weight"] = sd["stages.3.shortcut.conv.weight"][p_s2, :, :, :][:, p_s1, :, :]
    sd["stages.3.shortcut.norm.weight"] = sd["stages.3.shortcut.norm.weight"][p_s2]
    sd["stages.3.shortcut.norm.bias"] = sd["stages.3.shortcut.norm.bias"][p_s2]

    # 4. Stage 2 blocks (b = 4, 5)
    for b in range(4, 6):
        p_int = torch.from_numpy(p_blocks_int[b])
        prefix = f"stages.{b}."
        sd[f"{prefix}conv1.weight"] = sd[f"{prefix}conv1.weight"][p_int, :, :, :][:, p_s2, :, :]
        sd[f"{prefix}norm1.weight"] = sd[f"{prefix}norm1.weight"][p_int]
        sd[f"{prefix}norm1.bias"] = sd[f"{prefix}norm1.bias"][p_int]
        sd[f"{prefix}conv2.weight"] = sd[f"{prefix}conv2.weight"][p_s2, :, :, :][:, p_int, :, :]
        sd[f"{prefix}norm2.weight"] = sd[f"{prefix}norm2.weight"][p_s2]
        sd[f"{prefix}norm2.bias"] = sd[f"{prefix}norm2.bias"][p_s2]

    # 5. Block 6 (downsampling Stage 2 -> Stage 3)
    p_int6 = torch.from_numpy(p_blocks_int[6])
    sd["stages.6.conv1.weight"] = sd["stages.6.conv1.weight"][p_int6, :, :, :][:, p_s2, :, :]
    sd["stages.6.norm1.weight"] = sd["stages.6.norm1.weight"][p_int6]
    sd["stages.6.norm1.bias"] = sd["stages.6.norm1.bias"][p_int6]
    sd["stages.6.conv2.weight"] = sd["stages.6.conv2.weight"][p_s3, :, :, :][:, p_int6, :, :]
    sd["stages.6.norm2.weight"] = sd["stages.6.norm2.weight"][p_s3]
    sd["stages.6.norm2.bias"] = sd["stages.6.norm2.bias"][p_s3]
    sd["stages.6.shortcut.conv.weight"] = sd["stages.6.shortcut.conv.weight"][p_s3, :, :, :][:, p_s2, :, :]
    sd["stages.6.shortcut.norm.weight"] = sd["stages.6.shortcut.norm.weight"][p_s3]
    sd["stages.6.shortcut.norm.bias"] = sd["stages.6.shortcut.norm.bias"][p_s3]

    # 6. Stage 3 blocks (b = 7, 8)
    for b in range(7, 9):
        p_int = torch.from_numpy(p_blocks_int[b])
        prefix = f"stages.{b}."
        sd[f"{prefix}conv1.weight"] = sd[f"{prefix}conv1.weight"][p_int, :, :, :][:, p_s3, :, :]
        sd[f"{prefix}norm1.weight"] = sd[f"{prefix}norm1.weight"][p_int]
        sd[f"{prefix}norm1.bias"] = sd[f"{prefix}norm1.bias"][p_int]
        sd[f"{prefix}conv2.weight"] = sd[f"{prefix}conv2.weight"][p_s3, :, :, :][:, p_int, :, :]
        sd[f"{prefix}norm2.weight"] = sd[f"{prefix}norm2.weight"][p_s3]
        sd[f"{prefix}norm2.bias"] = sd[f"{prefix}norm2.bias"][p_s3]

    # 7. Final Linear head (input channels permuted by p_s3, output classes untouched)
    sd["fc.weight"] = sd["fc.weight"][:, p_s3]

    permuted.load_state_dict(sd)
    return permuted


@torch.no_grad()
def match_weights_resnet20(
    model_A: nn.Module,
    model_B: nn.Module,
    max_iters: int = 3
) -> Tuple[nn.Module, Dict[str, any]]:
    """Aligns neuron channels of model_B to match model_A via Git Re-Basin coordinate descent.
    
    Args:
        model_A: Reference model (anchor).
        model_B: Target model to permute.
        max_iters: Number of coordinate descent sweeps across permutations.
        
    Returns:
        Tuple of (permuted_model_B, permutation_info_dict).
    """
    sd_A = model_A.state_dict()
    current_B = deepcopy(model_B)

    # Initialize permutations as identity
    p_s1 = np.arange(16)
    p_s2 = np.arange(32)
    p_s3 = np.arange(64)
    p_blocks_int = {b: np.arange(16 if b < 3 else (32 if b < 6 else 64)) for b in range(9)}

    for iteration in range(max_iters):
        sd_B = current_B.state_dict()

        # --- A. Update internal block permutations ---
        for b in range(9):
            prefix = f"stages.{b}."
            C_int = 16 if b < 3 else (32 if b < 6 else 64)
            # Cost matrix C(i, j): correlation of channel j in B to channel i in A
            cost = np.zeros((C_int, C_int), dtype=np.float64)

            # 1. conv1 output: shape (C_int, C_in, 3, 3)
            wA1 = sd_A[f"{prefix}conv1.weight"].reshape(C_int, -1).cpu().numpy()
            wB1 = sd_B[f"{prefix}conv1.weight"].reshape(C_int, -1).cpu().numpy()
            cost += wA1 @ wB1.T

            # 2. norm1: shape (C_int,)
            nA1 = sd_A[f"{prefix}norm1.weight"].cpu().numpy()
            nB1 = sd_B[f"{prefix}norm1.weight"].cpu().numpy()
            cost += np.outer(nA1, nB1)

            # 3. conv2 input: shape (C_out, C_int, 3, 3)
            # Channel axis for input is axis 1, permute to axis 0 and flatten
            wA2_flat = sd_A[f"{prefix}conv2.weight"].permute(1, 0, 2, 3).reshape(C_int, -1).cpu().numpy()
            wB2_flat = sd_B[f"{prefix}conv2.weight"].permute(1, 0, 2, 3).reshape(C_int, -1).cpu().numpy()
            cost += wA2_flat @ wB2_flat.T

            p_blocks_int[b] = solve_permutation(cost)

        # Apply internal permutations
        current_B = apply_permutations_to_resnet20(model_B, p_s1, p_s2, p_s3, p_blocks_int)
        sd_B = current_B.state_dict()

        # --- B. Update Stage 1 residual stream (size 16) ---
        cost_s1 = np.zeros((16, 16), dtype=np.float64)
        # init_conv output
        cost_s1 += sd_A["init_conv.weight"].reshape(16, -1).cpu().numpy() @ sd_B["init_conv.weight"].reshape(16, -1).cpu().numpy().T
        cost_s1 += np.outer(sd_A["init_norm.weight"].cpu().numpy(), sd_B["init_norm.weight"].cpu().numpy())
        # Stage 1 conv1 inputs and conv2 outputs
        for b in range(3):
            prefix = f"stages.{b}."
            # conv1 input
            wA_in = sd_A[f"{prefix}conv1.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
            wB_in = sd_B[f"{prefix}conv1.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
            cost_s1 += wA_in @ wB_in.T
            # conv2 output
            wA_out = sd_A[f"{prefix}conv2.weight"].reshape(16, -1).cpu().numpy()
            wB_out = sd_B[f"{prefix}conv2.weight"].reshape(16, -1).cpu().numpy()
            cost_s1 += wA_out @ wB_out.T
            cost_s1 += np.outer(sd_A[f"{prefix}norm2.weight"].cpu().numpy(), sd_B[f"{prefix}norm2.weight"].cpu().numpy())
        # block 3 conv1 and shortcut input
        wA_b3in = sd_A["stages.3.conv1.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
        wB_b3in = sd_B["stages.3.conv1.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
        cost_s1 += wA_b3in @ wB_b3in.T
        wA_sc = sd_A["stages.3.shortcut.conv.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
        wB_sc = sd_B["stages.3.shortcut.conv.weight"].transpose(1, 0).reshape(16, -1).cpu().numpy()
        cost_s1 += wA_sc @ wB_sc.T

        p_s1 = solve_permutation(cost_s1)

        # Apply s1 permutation
        current_B = apply_permutations_to_resnet20(model_B, p_s1, p_s2, p_s3, p_blocks_int)
        sd_B = current_B.state_dict()

        # --- C. Update Stage 2 residual stream (size 32) ---
        cost_s2 = np.zeros((32, 32), dtype=np.float64)
        # block 3 outputs
        cost_s2 += sd_A["stages.3.conv2.weight"].reshape(32, -1).cpu().numpy() @ sd_B["stages.3.conv2.weight"].reshape(32, -1).cpu().numpy().T
        cost_s2 += np.outer(sd_A["stages.3.norm2.weight"].cpu().numpy(), sd_B["stages.3.norm2.weight"].cpu().numpy())
        cost_s2 += sd_A["stages.3.shortcut.conv.weight"].reshape(32, -1).cpu().numpy() @ sd_B["stages.3.shortcut.conv.weight"].reshape(32, -1).cpu().numpy().T
        cost_s2 += np.outer(sd_A["stages.3.shortcut.norm.weight"].cpu().numpy(), sd_B["stages.3.shortcut.norm.weight"].cpu().numpy())
        # blocks 4, 5
        for b in (4, 5):
            prefix = f"stages.{b}."
            wA_in = sd_A[f"{prefix}conv1.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
            wB_in = sd_B[f"{prefix}conv1.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
            cost_s2 += wA_in @ wB_in.T
            wA_out = sd_A[f"{prefix}conv2.weight"].reshape(32, -1).cpu().numpy()
            wB_out = sd_B[f"{prefix}conv2.weight"].reshape(32, -1).cpu().numpy()
            cost_s2 += wA_out @ wB_out.T
            cost_s2 += np.outer(sd_A[f"{prefix}norm2.weight"].cpu().numpy(), sd_B[f"{prefix}norm2.weight"].cpu().numpy())
        # block 6 inputs
        wA_b6in = sd_A["stages.6.conv1.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
        wB_b6in = sd_B["stages.6.conv1.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
        cost_s2 += wA_b6in @ wB_b6in.T
        wA_b6sc = sd_A["stages.6.shortcut.conv.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
        wB_b6sc = sd_B["stages.6.shortcut.conv.weight"].transpose(1, 0).reshape(32, -1).cpu().numpy()
        cost_s2 += wA_b6sc @ wB_b6sc.T

        p_s2 = solve_permutation(cost_s2)

        # Apply s2 permutation
        current_B = apply_permutations_to_resnet20(model_B, p_s1, p_s2, p_s3, p_blocks_int)
        sd_B = current_B.state_dict()

        # --- D. Update Stage 3 residual stream (size 64) ---
        cost_s3 = np.zeros((64, 64), dtype=np.float64)
        # block 6 outputs
        cost_s3 += sd_A["stages.6.conv2.weight"].reshape(64, -1).cpu().numpy() @ sd_B["stages.6.conv2.weight"].reshape(64, -1).cpu().numpy().T
        cost_s3 += np.outer(sd_A["stages.6.norm2.weight"].cpu().numpy(), sd_B["stages.6.norm2.weight"].cpu().numpy())
        cost_s3 += sd_A["stages.6.shortcut.conv.weight"].reshape(64, -1).cpu().numpy() @ sd_B["stages.6.shortcut.conv.weight"].reshape(64, -1).cpu().numpy().T
        cost_s3 += np.outer(sd_A["stages.6.shortcut.norm.weight"].cpu().numpy(), sd_B["stages.6.shortcut.norm.weight"].cpu().numpy())
        # blocks 7, 8
        for b in (7, 8):
            prefix = f"stages.{b}."
            wA_in = sd_A[f"{prefix}conv1.weight"].transpose(1, 0).reshape(64, -1).cpu().numpy()
            wB_in = sd_B[f"{prefix}conv1.weight"].transpose(1, 0).reshape(64, -1).cpu().numpy()
            cost_s3 += wA_in @ wB_in.T
            wA_out = sd_A[f"{prefix}conv2.weight"].reshape(64, -1).cpu().numpy()
            wB_out = sd_B[f"{prefix}conv2.weight"].reshape(64, -1).cpu().numpy()
            cost_s3 += wA_out @ wB_out.T
            cost_s3 += np.outer(sd_A[f"{prefix}norm2.weight"].cpu().numpy(), sd_B[f"{prefix}norm2.weight"].cpu().numpy())
        # fc input: shape (num_classes, 64)
        wA_fc = sd_A["fc.weight"].transpose(1, 0).cpu().numpy()
        wB_fc = sd_B["fc.weight"].transpose(1, 0).cpu().numpy()
        cost_s3 += wA_fc @ wB_fc.T

        p_s3 = solve_permutation(cost_s3)

        # Apply s3 permutation
        current_B = apply_permutations_to_resnet20(model_B, p_s1, p_s2, p_s3, p_blocks_int)

    # Compute permuted midpoint model
    rebasin_model = deepcopy(model_A)
    sd_rebasin = OrderedDict()
    sd_curr = current_B.state_dict()
    for k in sd_A.keys():
        if torch.is_floating_point(sd_A[k]):
            sd_rebasin[k] = 0.5 * (sd_A[k] + sd_curr[k])
        else:
            sd_rebasin[k] = sd_A[k]
    rebasin_model.load_state_dict(sd_rebasin)

    perm_info = {
        "p_stage1": p_s1.tolist(),
        "p_stage2": p_s2.tolist(),
        "p_stage3": p_s3.tolist(),
        "p_blocks_int": {b: p_blocks_int[b].tolist() for b in range(9)}
    }
    return current_B, rebasin_model, perm_info
