"""Centered Kernel Alignment (CKA) for representational similarity analysis.

Implements Linear CKA (Kornblith et al., ICML 2019) to quantify feature representation
alignment between neural networks across different stages and training conditions.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-10) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between two feature matrices.
    
    Args:
        X: Feature matrix of shape (N, D1).
        Y: Feature matrix of shape (N, D2).
        eps: Small constant to avoid division by zero.
        
    Returns:
        Linear CKA similarity score in [0.0, 1.0].
    """
    if X.dim() > 2:
        X = X.flatten(1)
    if Y.dim() > 2:
        Y = Y.flatten(1)

    assert X.size(0) == Y.size(0), f"Sample counts must match: {X.size(0)} vs {Y.size(0)}"
    n = X.size(0)
    if n <= 1:
        return 1.0

    X = X.to(torch.float64)
    Y = Y.to(torch.float64)

    # Center columns
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    # Use cross-covariance formulation if D is small, Gram matrix if N is small
    d1, d2 = X_c.size(1), Y_c.size(1)
    if d1 * d2 < n * n:
        # Cross-covariance formulation: ||Y_c^T X_c||_F^2
        cov_xy = torch.matmul(Y_c.t(), X_c)
        hsic_xy = torch.sum(cov_xy ** 2)
        
        cov_xx = torch.matmul(X_c.t(), X_c)
        hsic_xx = torch.sum(cov_xx ** 2)
        
        cov_yy = torch.matmul(Y_c.t(), Y_c)
        hsic_yy = torch.sum(cov_yy ** 2)
    else:
        # Gram matrix formulation: tr(K_c L_c)
        K = torch.matmul(X_c, X_c.t())
        L = torch.matmul(Y_c, Y_c.t())
        hsic_xy = torch.sum(K * L)
        hsic_xx = torch.sum(K * K)
        hsic_yy = torch.sum(L * L)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom <= eps:
        return 0.0
    cka = (hsic_xy / denom).item()
    return float(max(0.0, min(1.0, cka)))


@torch.no_grad()
def extract_stage_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_samples: int = 1000
) -> Dict[str, torch.Tensor]:
    """Extracts intermediate representations for ResNet20LN across architectural stages.
    
    Extracted stages:
    - 'init': Post-init conv + LayerNorm + ReLU (channels=16)
    - 'stage1': Output of Stage 1 (after block 2, channels=16)
    - 'stage2': Output of Stage 2 (after block 5, channels=32)
    - 'stage3': Output of Stage 3 (after block 8, channels=64)
    - 'penultimate': Global average pooled representation (dim=64)
    - 'logits': Pre-softmax output (dim=10)
    """
    model.eval()
    model.to(device)

    collected: Dict[str, List[torch.Tensor]] = {
        "init": [],
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "penultimate": [],
        "logits": []
    }
    total_samples = 0

    for x, _ in dataloader:
        if total_samples >= max_samples:
            break
        x = x.to(device)
        batch_size = x.size(0)
        if total_samples + batch_size > max_samples:
            x = x[:max_samples - total_samples]
            batch_size = x.size(0)

        # Stage-by-stage forward pass
        out_init = model.init_act(model.init_norm(model.init_conv(x)))
        
        # Stage 1: blocks 0, 1, 2
        s1 = out_init
        for b in range(3):
            s1 = model.stages[b](s1)
            
        # Stage 2: blocks 3, 4, 5
        s2 = s1
        for b in range(3, 6):
            s2 = model.stages[b](s2)
            
        # Stage 3: blocks 6, 7, 8
        s3 = s2
        for b in range(6, 9):
            s3 = model.stages[b](s3)
            
        pooled = F.adaptive_avg_pool2d(s3, (1, 1)).flatten(1)
        logits = model.fc(pooled)

        # Apply global pooling to spatial activations to compare feature representations
        collected["init"].append(F.adaptive_avg_pool2d(out_init, (1, 1)).flatten(1).cpu())
        collected["stage1"].append(F.adaptive_avg_pool2d(s1, (1, 1)).flatten(1).cpu())
        collected["stage2"].append(F.adaptive_avg_pool2d(s2, (1, 1)).flatten(1).cpu())
        collected["stage3"].append(pooled.cpu())
        collected["penultimate"].append(pooled.cpu())
        collected["logits"].append(logits.cpu())

        total_samples += batch_size

    # Concatenate across batches
    return {k: torch.cat(v, dim=0) for k, v in collected.items()}


def compute_pairwise_cka_matrix(
    features_A: Dict[str, torch.Tensor],
    features_B: Dict[str, torch.Tensor],
    stages: Optional[List[str]] = None
) -> Tuple[List[str], torch.Tensor]:
    """Computes full stage-by-stage CKA similarity matrix between two models."""
    if stages is None:
        stages = ["init", "stage1", "stage2", "stage3", "penultimate", "logits"]

    m = len(stages)
    matrix = torch.zeros((m, m), dtype=torch.float32)

    for i, s_A in enumerate(stages):
        for j, s_B in enumerate(stages):
            matrix[i, j] = linear_cka(features_A[s_A], features_B[s_B])

    return stages, matrix


def compute_stage_cka_summary(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_samples: int = 1000
) -> Dict[str, any]:
    """Computes diagonal and pairwise CKA representation similarity between two models."""
    feat_A = extract_stage_features(model_A, dataloader, device=device, max_samples=max_samples)
    feat_B = extract_stage_features(model_B, dataloader, device=device, max_samples=max_samples)

    stages = ["init", "stage1", "stage2", "stage3", "penultimate", "logits"]
    diag_cka = {}
    for s in stages:
        diag_cka[s] = linear_cka(feat_A[s], feat_B[s])

    stages, full_matrix = compute_pairwise_cka_matrix(feat_A, feat_B, stages=stages)

    return {
        "diagonal_cka": diag_cka,
        "stages": stages,
        "matrix": full_matrix.tolist()
    }
