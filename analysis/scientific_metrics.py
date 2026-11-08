"""
Scientific Metrics & Representation Decomposition Engine.
Implements the exact metric suite from Frozen Plan (v1.0):
1. Geometric: 11-point loss barrier (Sharma et al.), D_pair (body/head split), directional curvature.
2. Representational: Angular CKA with reduced-bias HSIC estimator (Lange et al.), feature velocity.
3. Functional Decomposition:
   - Linear probe on frozen features (Probe Gap - upper bound on head fix)
   - Few-shot head-only fine-tuning on 200 balanced samples (Few-shot Head Gain)
   - Head Share = Probe Gap / (No-Branch Acc - Merged Acc)
   - Cumulative partial merge profile (ZipIt-style stage prefixes j=0..4 + ensembled suffixes)
   - Logit-offset fidelity: regress Delta Logits on log(p_A / p_B) for H-B2.
"""

from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_parameter_distances(
    model_A: nn.Module,
    model_B: nn.Module,
    model_origin: Optional[nn.Module] = None
) -> Dict[str, float]:
    """Computes normalized Euclidean distance D_pair = ||w_A - w_B|| / sqrt(P),
    split into body and head components, and D_origin.
    """
    total_sq_diff = 0.0
    body_sq_diff = 0.0
    head_sq_diff = 0.0
    total_params = 0
    body_params = 0
    head_params = 0

    p_A = dict(model_A.named_parameters())
    p_B = dict(model_B.named_parameters())

    for name in p_A:
        if name not in p_B:
            continue
        diff = (p_A[name].data - p_B[name].data).view(-1)
        sq_diff = torch.sum(diff ** 2).item()
        n_p = diff.numel()

        total_sq_diff += sq_diff
        total_params += n_p

        # Head is defined as the final linear classifier ('fc' or 'classifier')
        if "fc" in name or "classifier" in name or "linear" in name:
            head_sq_diff += sq_diff
            head_params += n_p
        else:
            body_sq_diff += sq_diff
            body_params += n_p

    d_pair = math.sqrt(total_sq_diff) / math.sqrt(max(total_params, 1))
    d_body = math.sqrt(body_sq_diff) / math.sqrt(max(body_params, 1))
    d_head = math.sqrt(head_sq_diff) / math.sqrt(max(head_params, 1))

    results = {
        "d_pair": float(d_pair),
        "d_body": float(d_body),
        "d_head": float(d_head),
        "raw_l2": float(math.sqrt(total_sq_diff)),
        "num_params": total_params,
        "num_body_params": body_params,
        "num_head_params": head_params,
    }

    if model_origin is not None:
        p_0 = dict(model_origin.named_parameters())
        orig_sq_diff = 0.0
        for name in p_A:
            if name in p_0:
                mid = 0.5 * (p_A[name].data + p_B[name].data)
                orig_sq_diff += torch.sum((mid - p_0[name].data) ** 2).item()
        results["d_origin"] = float(math.sqrt(orig_sq_diff) / math.sqrt(max(total_params, 1)))

    return results


def interpolate_state_dicts(
    sd_A: Dict[str, torch.Tensor],
    sd_B: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """Computes linear interpolation W(alpha) = (1 - alpha) * W_A + alpha * W_B."""
    sd_interp = {}
    for k in sd_A:
        if k in sd_B:
            if sd_A[k].dtype.is_floating_point:
                sd_interp[k] = (1.0 - alpha) * sd_A[k] + alpha * sd_B[k]
            else:
                sd_interp[k] = sd_A[k] if alpha < 0.5 else sd_B[k]
    return sd_interp


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates cross-entropy loss and top-1 accuracy on a dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    mean_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return mean_loss, acc


def compute_interpolated_barriers(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_points: int = 11
) -> Dict[str, any]:
    """Evaluates 11-point linear interpolation path under Sharma et al. definition."""
    alphas = np.linspace(0.0, 1.0, num_points)
    sd_A = model_A.state_dict()
    sd_B = model_B.state_dict()

    eval_model = deepcopy(model_A).to(device)

    losses = []
    accuracies = []

    for alpha in alphas:
        interp_sd = interpolate_state_dicts(sd_A, sd_B, float(alpha))
        eval_model.load_state_dict(interp_sd)
        loss, acc = evaluate_model(eval_model, dataloader, device)
        losses.append(loss)
        accuracies.append(acc)

    loss_A, loss_B = losses[0], losses[-1]
    acc_A, acc_B = accuracies[0], accuracies[-1]

    # Endpoint-interpolated loss barrier: max_alpha L(W(alpha)) - [(1 - alpha) * L_A + alpha * L_B]
    loss_barriers = [
        losses[i] - ((1.0 - alphas[i]) * loss_A + alphas[i] * loss_B)
        for i in range(num_points)
    ]
    max_loss_barrier = float(max(loss_barriers))

    # Accuracy dip: max_alpha [(1 - alpha) * Acc_A + alpha * Acc_B] - Acc(W(alpha))
    acc_dips = [
        ((1.0 - alphas[i]) * acc_A + alphas[i] * acc_B) - accuracies[i]
        for i in range(num_points)
    ]
    max_acc_dip = float(max(acc_dips))

    # Midpoint metrics (alpha = 0.5)
    mid_idx = num_points // 2
    merged_loss = float(losses[mid_idx])
    merged_acc = float(accuracies[mid_idx])

    return {
        "alphas": alphas.tolist(),
        "losses": losses,
        "accuracies": accuracies,
        "loss_barriers": loss_barriers,
        "acc_dips": acc_dips,
        "max_loss_barrier": max_loss_barrier,
        "max_acc_dip": max_acc_dip,
        "merged_loss": merged_loss,
        "merged_acc": merged_acc,
        "acc_A": float(acc_A),
        "acc_B": float(acc_B),
    }


def compute_directional_curvature(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epsilon: float = 1e-3
) -> Dict[str, float]:
    """Computes directional loss curvature along body and head components of v = theta_B - theta_A
    at the midpoint W_mid using second-order finite differences:
    kappa = [L(W_mid + eps * v) - 2 * L(W_mid) + L(W_mid - eps * v)] / eps^2.
    """
    eval_model = deepcopy(model_A).to(device)
    sd_A = model_A.state_dict()
    sd_B = model_B.state_dict()
    sd_mid = interpolate_state_dicts(sd_A, sd_B, 0.5)

    eval_model.load_state_dict(sd_mid)
    loss_mid, _ = evaluate_model(eval_model, dataloader, device)

    # 1. Total Curvature
    sd_pos = {}
    sd_neg = {}
    for k in sd_mid:
        if sd_mid[k].dtype.is_floating_point:
            v_k = sd_B[k] - sd_A[k]
            sd_pos[k] = sd_mid[k] + epsilon * v_k
            sd_neg[k] = sd_mid[k] - epsilon * v_k
        else:
            sd_pos[k] = sd_mid[k]
            sd_neg[k] = sd_mid[k]

    eval_model.load_state_dict(sd_pos)
    loss_pos, _ = evaluate_model(eval_model, dataloader, device)
    eval_model.load_state_dict(sd_neg)
    loss_neg, _ = evaluate_model(eval_model, dataloader, device)
    kappa_total = (loss_pos - 2.0 * loss_mid + loss_neg) / (epsilon ** 2)

    # 2. Body-only Curvature
    sd_pos_body = {}
    sd_neg_body = {}
    for k in sd_mid:
        if sd_mid[k].dtype.is_floating_point and not ("fc" in k or "classifier" in k):
            v_k = sd_B[k] - sd_A[k]
            sd_pos_body[k] = sd_mid[k] + epsilon * v_k
            sd_neg_body[k] = sd_mid[k] - epsilon * v_k
        else:
            sd_pos_body[k] = sd_mid[k]
            sd_neg_body[k] = sd_mid[k]

    eval_model.load_state_dict(sd_pos_body)
    loss_pos_b, _ = evaluate_model(eval_model, dataloader, device)
    eval_model.load_state_dict(sd_neg_body)
    loss_neg_b, _ = evaluate_model(eval_model, dataloader, device)
    kappa_body = (loss_pos_b - 2.0 * loss_mid + loss_neg_b) / (epsilon ** 2)

    # 3. Head-only Curvature
    sd_pos_head = {}
    sd_neg_head = {}
    for k in sd_mid:
        if sd_mid[k].dtype.is_floating_point and ("fc" in k or "classifier" in k):
            v_k = sd_B[k] - sd_A[k]
            sd_pos_head[k] = sd_mid[k] + epsilon * v_k
            sd_neg_head[k] = sd_mid[k] - epsilon * v_k
        else:
            sd_pos_head[k] = sd_mid[k]
            sd_neg_head[k] = sd_mid[k]

    eval_model.load_state_dict(sd_pos_head)
    loss_pos_h, _ = evaluate_model(eval_model, dataloader, device)
    eval_model.load_state_dict(sd_neg_head)
    loss_neg_h, _ = evaluate_model(eval_model, dataloader, device)
    kappa_head = (loss_pos_h - 2.0 * loss_mid + loss_neg_h) / (epsilon ** 2)

    return {
        "curvature_total": float(kappa_total),
        "curvature_body": float(kappa_body),
        "curvature_head": float(kappa_head)
    }


def compute_unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Computes the reduced-bias HSIC estimator of Lange et al. (TAG-ML 2023) / Song et al.
    K and L are n x n Gram matrices.
    """
    n = K.shape[0]
    if n < 4:
        # Fall back to standard centered HSIC for tiny batches
        H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
        return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    # Zero out diagonal
    K_tilde = K.clone()
    K_tilde.fill_diagonal_(0.0)
    L_tilde = L.clone()
    L_tilde.fill_diagonal_(0.0)

    term1 = torch.sum(K_tilde * L_tilde)
    term2 = (torch.sum(K_tilde, dim=1) @ torch.sum(L_tilde, dim=1)) / (n - 2)
    term3 = (torch.sum(K_tilde) * torch.sum(L_tilde)) / ((n - 1) * (n - 2))

    hsic = (term1 + term3 - 2.0 * term2) / (n * (n - 3))
    return hsic


def compute_angular_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Computes Angular CKA on representations X and Y:
    Linear CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    Angular CKA = 1 - (2 / pi) * arccos(clamp(CKA, -1, 1)).
    """
    X = X.view(X.shape[0], -1).float()
    Y = Y.view(Y.shape[0], -1).float()

    # Center representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic_xy = compute_unbiased_hsic(K, L)
    hsic_xx = compute_unbiased_hsic(K, K)
    hsic_yy = compute_unbiased_hsic(L, L)

    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12))
    cka = (hsic_xy / denom).item()
    cka_clamped = max(-1.0, min(1.0, cka))

    # Angular distance transform
    angular_cka = 1.0 - (2.0 / math.pi) * math.acos(cka_clamped)
    return float(angular_cka)


def extract_penultimate_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts features before the final linear classifier."""
    model.eval()
    features_list = []
    targets_list = []

    def hook_fn(module, input, output):
        features_list.append(input[0].detach().cpu())

    # Attach hook to fc layer
    hook = None
    for name, module in model.named_modules():
        if "fc" in name or "classifier" in name:
            hook = module.register_forward_hook(hook_fn)
            break

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _ = model(x)
            targets_list.append(y.cpu())

    if hook is not None:
        hook.remove()

    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return features, targets


def evaluate_linear_probe_gap(
    model_merged: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.05
) -> Tuple[float, float]:
    """Trains a linear probe on frozen features from full balanced train set.
    Returns: probe_acc, probe_gap = probe_acc - merged_acc.
    """
    _, merged_acc = evaluate_model(model_merged, test_loader, device)

    # Extract features
    X_train, y_train = extract_penultimate_features(model_merged, train_loader, device)
    X_test, y_test = extract_penultimate_features(model_merged, test_loader, device)

    in_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    probe = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for bx, by in loader_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = probe(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_out = probe(X_test.to(device))
        probe_acc = (test_out.argmax(dim=1).cpu() == y_test).float().mean().item()

    probe_gap = float(probe_acc - merged_acc)
    return float(probe_acc), probe_gap


def evaluate_few_shot_head_gain(
    model_merged: nn.Module,
    balanced_few_shot_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.01
) -> Tuple[float, float]:
    """Fine-tunes only the linear head on 200 balanced samples (20 per class, 5 epochs).
    Returns: fine_tuned_acc, few_shot_gain = fine_tuned_acc - merged_acc.
    """
    _, merged_acc = evaluate_model(model_merged, test_loader, device)

    tuned_model = deepcopy(model_merged).to(device)
    # Freeze body
    for name, p in tuned_model.named_parameters():
        if not ("fc" in name or "classifier" in name):
            p.requires_grad = False
        else:
            p.requires_grad = True

    head_params = [p for p in tuned_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(head_params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    tuned_model.train()
    for _ in range(epochs):
        for x, y in balanced_few_shot_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = tuned_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    _, fine_tuned_acc = evaluate_model(tuned_model, test_loader, device)
    gain = float(fine_tuned_acc - merged_acc)
    return float(fine_tuned_acc), gain


def compute_head_share(probe_gap: float, no_branch_acc: float, merged_acc: float) -> float:
    """Head share = probe gap / (no-branch accuracy - merged accuracy)."""
    denom = max(no_branch_acc - merged_acc, 1e-6)
    return float(np.clip(probe_gap / denom, 0.0, 1.0))


def compute_logit_offset_fidelity(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    p_A: np.ndarray,
    p_B: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """Tests Menon et al. logit-adjustment prediction (H-B2):
    Delta Logit_c ~ log(p_A(c) / p_B(c)).
    """
    model_A.eval()
    model_B.eval()

    num_classes = len(p_A)
    sum_logits_A = torch.zeros(num_classes, device=device)
    sum_logits_B = torch.zeros(num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out_A = model_A(x)
            out_B = model_B(x)
            for c in range(num_classes):
                mask = (y == c)
                if mask.any():
                    sum_logits_A[c] += out_A[mask, c].sum()
                    sum_logits_B[c] += out_B[mask, c].sum()
                    class_counts[c] += mask.sum()

    mean_logits_A = (sum_logits_A / torch.clamp(class_counts, min=1)).cpu().numpy()
    mean_logits_B = (sum_logits_B / torch.clamp(class_counts, min=1)).cpu().numpy()
    delta_logits = mean_logits_A - mean_logits_B

    # Avoid div by zero for absent classes (s=1.0)
    eps = 1e-6
    valid_mask = (p_A > eps) & (p_B > eps)
    if valid_mask.sum() < 3:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0}

    log_ratio = np.log(p_A[valid_mask] / p_B[valid_mask])
    y_vals = delta_logits[valid_mask]

    # OLS fit
    A_mat = np.vstack([log_ratio, np.ones(len(log_ratio))]).T
    slope, intercept = np.linalg.lstsq(A_mat, y_vals, rcond=None)[0]

    y_pred = slope * log_ratio + intercept
    sst = np.sum((y_vals - np.mean(y_vals)) ** 2)
    ssr = np.sum((y_vals - y_pred) ** 2)
    r2 = 1.0 - (ssr / max(sst, 1e-12))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(max(0.0, r2))
    }


def evaluate_cumulative_partial_merges(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates cumulative partial merge profile for j=0..4:
    Averages layers up to stage j, and ensembles the remaining layers' logits.
    Stages:
    0: init_conv
    1: stage 1 (layer1)
    2: stage 2 (layer2)
    3: stage 3 (layer3)
    4: full network (fc included)
    """
    model_A.eval()
    model_B.eval()

    stage_modules = ["init", "stage1", "stage2", "stage3", "fc"]
    results = {}

    for j in range(len(stage_modules)):
        # Determine which parameters belong to merged prefix (<= j) vs separate suffixes (> j)
        sd_merged_prefix_A = model_A.state_dict()
        sd_merged_prefix_B = model_B.state_dict()
        
        merged_sd = {}
        for k in sd_merged_prefix_A:
            # Map parameter name to stage index
            if "conv1" in k or "norm1" in k:
                param_stage = 0
            elif "layer1" in k:
                param_stage = 1
            elif "layer2" in k:
                param_stage = 2
            elif "layer3" in k:
                param_stage = 3
            elif "fc" in k or "classifier" in k:
                param_stage = 4
            else:
                param_stage = 0

            if param_stage <= j:
                # Averaged prefix
                if sd_merged_prefix_A[k].dtype.is_floating_point:
                    merged_sd[k] = 0.5 * (sd_merged_prefix_A[k] + sd_merged_prefix_B[k])
                else:
                    merged_sd[k] = sd_merged_prefix_A[k]
            else:
                # Suffix remains model-specific
                pass

        # Build two models with merged prefix + individual suffix
        model_A_hybrid = deepcopy(model_A).to(device)
        model_B_hybrid = deepcopy(model_B).to(device)

        sd_A_curr = model_A_hybrid.state_dict()
        sd_B_curr = model_B_hybrid.state_dict()

        for k in merged_sd:
            sd_A_curr[k] = merged_sd[k]
            sd_B_curr[k] = merged_sd[k]

        model_A_hybrid.load_state_dict(sd_A_curr)
        model_B_hybrid.load_state_dict(sd_B_curr)

        model_A_hybrid.eval()
        model_B_hybrid.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits_A = model_A_hybrid(x)
                logits_B = model_B_hybrid(x)
                ens_logits = 0.5 * (logits_A + logits_B)
                pred = ens_logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / max(total, 1)
        results[f"stage_prefix_{j}_{stage_modules[j]}"] = float(acc)

    return results


def extract_stage_representations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Extracts internal activations across ResNet-20LN stages:
    0: init_conv (post-ReLU)
    1: stage1 (after 3 blocks)
    2: stage2 (after 6 blocks)
    3: stage3 (after 9 blocks)
    4: penultimate (global average pooled feature vector)
    5: head (output logits)
    """
    model.eval()
    acts_stage0 = []
    acts_stage1 = []
    acts_stage2 = []
    acts_stage3 = []
    acts_penultimate = []
    acts_head = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            out0 = model.init_act(model.init_norm(model.init_conv(x)))
            out1 = model.stages[:3](out0)
            out2 = model.stages[3:6](out1)
            out3 = model.stages[6:9](out2)
            out_pool = F.adaptive_avg_pool2d(out3, (1, 1)).flatten(1)
            out_logits = model.fc(out_pool)

            acts_stage0.append(out0.cpu())
            acts_stage1.append(out1.cpu())
            acts_stage2.append(out2.cpu())
            acts_stage3.append(out3.cpu())
            acts_penultimate.append(out_pool.cpu())
            acts_head.append(out_logits.cpu())

    return {
        "stage0_init": torch.cat(acts_stage0, dim=0),
        "stage1_residual": torch.cat(acts_stage1, dim=0),
        "stage2_residual": torch.cat(acts_stage2, dim=0),
        "stage3_residual": torch.cat(acts_stage3, dim=0),
        "penultimate_pooled": torch.cat(acts_penultimate, dim=0),
        "head_logits": torch.cat(acts_head, dim=0)
    }


def compute_layerwise_cka(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Computes Angular CKA across all 5 architectural depths (Hypothesis H-B1)."""
    acts_A = extract_stage_representations(model_A, dataloader, device)
    acts_B = extract_stage_representations(model_B, dataloader, device)

    cka_dict = {}
    for stage_name in acts_A:
        cka_val = compute_angular_cka(acts_A[stage_name], acts_B[stage_name])
        cka_dict[stage_name] = float(cka_val)

    return cka_dict


def evaluate_zero_shot_logit_adjustment(
    model: nn.Module,
    dataloader: DataLoader,
    p_mix: np.ndarray,
    device: torch.device,
    taus: Optional[List[float]] = None
) -> Dict[str, any]:
    """Evaluates Menon et al. zero-shot logit adjustment on model outputs (Hypothesis H-B2):
    adjusted_logits = raw_logits - tau * log(C * p_mix).
    No raw data is shared; requires only scalar class frequencies.
    """
    if taus is None:
        taus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            all_logits.append(out.cpu())
            all_targets.append(y)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    num_classes = len(p_mix)
    p_mix_clamped = np.clip(p_mix, 1e-6, 1.0)
    p_mix_tensor = torch.tensor(p_mix_clamped, dtype=torch.float32)
    log_offset = torch.log(num_classes * p_mix_tensor).view(1, num_classes)

    base_preds = logits.argmax(dim=1)
    base_acc = float((base_preds == targets).float().mean().item())

    best_tau = 1.0
    best_acc = base_acc
    tau_sweep = {}

    for tau in taus:
        adj_logits = logits - tau * log_offset
        preds = adj_logits.argmax(dim=1)
        acc = float((preds == targets).float().mean().item())
        tau_sweep[f"tau_{tau:.2f}"] = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    return {
        "base_acc": float(base_acc),
        "best_tau": float(best_tau),
        "best_adjusted_acc": float(best_acc),
        "gain": float(best_acc - base_acc),
        "tau_sweep": tau_sweep
    }


def evaluate_decoupled_head_ensemble(
    model_A: nn.Module,
    model_B: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    p_A: Optional[np.ndarray] = None,
    p_B: Optional[np.ndarray] = None,
    tau: float = 1.0
) -> Dict[str, float]:
    """Evaluates Cure 3: Body-only interpolation with decoupled client classification heads.
    1. Direct head ensembling (Softmax probability averaging).
    2. Prior-adjusted head ensembling (Menon-calibrated Softmax probabilities).
    """
    model_A.eval()
    model_B.eval()

    sd_A = model_A.state_dict()
    sd_B = model_B.state_dict()

    # Create merged body dictionary
    sd_body_merged = {}
    for k in sd_A:
        if not ("fc" in k or "classifier" in k):
            if sd_A[k].dtype.is_floating_point:
                sd_body_merged[k] = 0.5 * (sd_A[k] + sd_B[k])
            else:
                sd_body_merged[k] = sd_A[k]

    # Model with merged body + Head A
    model_hybrid_A = deepcopy(model_A).to(device)
    sd_hyb_A = model_hybrid_A.state_dict()
    for k in sd_body_merged:
        sd_hyb_A[k] = sd_body_merged[k]
    model_hybrid_A.load_state_dict(sd_hyb_A)
    model_hybrid_A.eval()

    # Model with merged body + Head B
    model_hybrid_B = deepcopy(model_B).to(device)
    sd_hyb_B = model_hybrid_B.state_dict()
    for k in sd_body_merged:
        sd_hyb_B[k] = sd_body_merged[k]
    model_hybrid_B.load_state_dict(sd_hyb_B)
    model_hybrid_B.eval()

    # Full naive merged model
    model_full_merged = deepcopy(model_A).to(device)
    sd_full = {}
    for k in sd_A:
        if sd_A[k].dtype.is_floating_point:
            sd_full[k] = 0.5 * (sd_A[k] + sd_B[k])
        else:
            sd_full[k] = sd_A[k]
    model_full_merged.load_state_dict(sd_full)
    model_full_merged.eval()

    correct_naive = 0
    correct_prob_ens = 0
    correct_adj_prob_ens = 0
    total = 0

    has_priors = (p_A is not None and p_B is not None)
    if has_priors:
        num_classes = len(p_A)
        eps = 1e-6
        offset_A = torch.tensor(np.log(num_classes * np.clip(p_A, eps, 1.0)), dtype=torch.float32, device=device).view(1, num_classes)
        offset_B = torch.tensor(np.log(num_classes * np.clip(p_B, eps, 1.0)), dtype=torch.float32, device=device).view(1, num_classes)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out_naive = model_full_merged(x)
            out_A = model_hybrid_A(x)
            out_B = model_hybrid_B(x)

            pred_naive = out_naive.argmax(dim=1)
            correct_naive += (pred_naive == y).sum().item()

            prob_A = F.softmax(out_A, dim=1)
            prob_B = F.softmax(out_B, dim=1)
            prob_ens = 0.5 * (prob_A + prob_B)
            pred_prob = prob_ens.argmax(dim=1)
            correct_prob_ens += (pred_prob == y).sum().item()

            if has_priors:
                adj_out_A = out_A - tau * offset_A
                adj_out_B = out_B - tau * offset_B
                adj_prob_A = F.softmax(adj_out_A, dim=1)
                adj_prob_B = F.softmax(adj_out_B, dim=1)
                adj_prob_ens = 0.5 * (adj_prob_A + adj_prob_B)
                pred_adj_prob = adj_prob_ens.argmax(dim=1)
                correct_adj_prob_ens += (pred_adj_prob == y).sum().item()

            total += y.size(0)

    n = max(total, 1)
    results = {
        "naive_merged_acc": float(correct_naive / n),
        "decoupled_prob_ens_acc": float(correct_prob_ens / n),
    }
    if has_priors:
        results["decoupled_adj_prob_ens_acc"] = float(correct_adj_prob_ens / n)

    return results


