"""
Science Harness for Onset-Dependent Mergeability under Persistent Divergence.
Implements the core science protocol from Frozen Plan (v1.0):
- Centralized SGD trunk on balanced CIFAR-10 (momentum 0.9, lr 0.1, batch 128, no weight decay).
- Warm-up-stable-decay schedule (2% linear warm-up, constant lr, 10% linear decay before merge).
- Exact branch cloning with momentum buffer copying.
- Checkpointing every 1% for Matched Design (equal D_pair +/- 5%).
- Complete metric evaluation: 11-point barrier, body/head D_pair, curvature, Angular CKA,
  linear probe gap, few-shot head gain, head share, cumulative ZipIt prefixes, logit offset fidelity.
"""

import argparse
from copy import deepcopy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_outage.data.cifar import load_cifar10
from fl_outage.data.scientific_partitions import get_scientific_subsets, create_scientific_partition_indices
from fl_outage.models.resnet import ResNet20LN
from analysis.scientific_metrics import (
    compute_parameter_distances,
    compute_interpolated_barriers,
    compute_directional_curvature,
    compute_angular_cka,
    extract_penultimate_features,
    evaluate_linear_probe_gap,
    evaluate_few_shot_head_gain,
    compute_head_share,
    evaluate_cumulative_partial_merges,
    compute_logit_offset_fidelity,
    evaluate_model
)


class WarmupStableDecayScheduler:
    """Implements 2% linear warm-up, constant lr, and 10% linear decay before measurement."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int,
                 warmup_frac: float = 0.02, decay_frac: float = 0.10):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_frac * total_steps)
        self.decay_steps = int(decay_frac * total_steps)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr * (self.current_step / max(self.warmup_steps, 1))
        elif self.current_step <= self.warmup_steps + self.stable_steps:
            # Stable constant lr
            lr = self.base_lr
        else:
            # Linear decay
            decay_progress = (self.current_step - self.warmup_steps - self.stable_steps) / max(self.decay_steps, 1)
            lr = self.base_lr * max(0.0, 1.0 - decay_progress)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def copy_optimizer_state(src_opt: torch.optim.Optimizer, tgt_opt: torch.optim.Optimizer, device: torch.device):
    """Deep-copies optimizer state dict including momentum buffers."""
    tgt_opt.load_state_dict(deepcopy(src_opt.state_dict()))
    for state in tgt_opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def build_probe_set(test_dataset, num_samples: int = 1000, seed: int = 42) -> DataLoader:
    """Builds fixed probe set for Angular CKA and feature velocity tracking."""
    rng = np.random.RandomState(seed)
    targets = np.array(test_dataset.targets)
    probe_indices = []
    classes = np.unique(targets)
    per_class = num_samples // len(classes)
    for c in classes:
        c_idx = np.where(targets == c)[0]
        chosen = rng.choice(c_idx, size=per_class, replace=False)
        probe_indices.extend(chosen)
    probe_subset = Subset(test_dataset, probe_indices)
    return DataLoader(probe_subset, batch_size=128, shuffle=False)


def run_science_experiment(
    total_steps: int = 25000,
    onset_pct: float = 2.0,
    source: str = "R3-L",
    s: float = 0.8,
    asymmetric_mode: Optional[str] = None,
    corruption_level: Optional[str] = None,
    seed: int = 42,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./runs/science_harness",
    data_dir: str = "./data/cache",
    save_checkpoints: bool = True,
    save_models: bool = False
) -> Dict[str, any]:
    """Runs a single science harness trial from trunk to branch merge."""
    device = torch.device(device_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    corruption_tag = f" | Corruption={corruption_level}" if corruption_level else ""
    print(f"\n" + "=" * 80)
    print(f"SCIENCE HARNESS: Source={source} | s={s}{corruption_tag} | Onset={onset_pct}% | Seed={seed} | Device={device_name}")
    print("=" * 80)

    # 1. Load CIFAR-10 data
    train_set, test_set = load_cifar10(root=data_dir, download=True)
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = (device.type == "cuda")

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    probe_loader = build_probe_set(test_set, num_samples=1000, seed=42)

    # 2. Partitions for branches
    subset_A, subset_B = get_scientific_subsets(
        train_dataset=train_set,
        source=source,
        s=s,
        asymmetric_mode=asymmetric_mode,
        corruption_level=corruption_level,
        seed=seed
    )

    loader_trunk = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    loader_A = DataLoader(subset_A, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    # If R2a, use a different generator seed for branch B DataLoader shuffle
    gen_B = torch.Generator().manual_seed(seed + 1000 if source == "R2a" else seed)
    loader_B = DataLoader(subset_B, batch_size=128, shuffle=True, generator=gen_B, num_workers=num_workers, pin_memory=pin_memory)

    # Few-shot loader (200 balanced samples: 20 per class)
    few_shot_indices = []
    rng_fs = np.random.RandomState(seed)
    targets_train = np.array(train_set.targets)
    for c in range(10):
        c_idx = np.where(targets_train == c)[0]
        few_shot_indices.extend(rng_fs.choice(c_idx, size=20, replace=False))
    few_shot_loader = DataLoader(Subset(train_set, few_shot_indices), batch_size=64, shuffle=True)

    # 3. Model & Optimizer initialization
    model = ResNet20LN(num_classes=10).to(device)
    model_origin = deepcopy(model).to(device)

    base_lr = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0)

    onset_step = int((onset_pct / 100.0) * total_steps)
    branch_steps = total_steps - onset_step

    # 4. Train Trunk up to Onset Step
    trunk_scheduler = WarmupStableDecayScheduler(
        optimizer, base_lr=base_lr, total_steps=total_steps, warmup_frac=0.02, decay_frac=0.10
    )

    step = 0
    trunk_iter = iter(loader_trunk)
    print(f"--> Training trunk for {onset_step} steps (Warm-up 2% = {int(0.02 * total_steps)} steps)...")

    while step < onset_step:
        try:
            x, y = next(trunk_iter)
        except StopIteration:
            trunk_iter = iter(loader_trunk)
            x, y = next(trunk_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        trunk_scheduler.step()
        step += 1

    checkpoint_onset = deepcopy(model)
    _, acc_onset = evaluate_model(model, test_loader, device)
    print(f"    Trunk reached step {step} | Test Acc = {acc_onset * 100.0:.2f}%")

    # 5. Branching: Clone Model A and Model B with momentum buffers
    print("--> Branching into Branch A and Branch B (Cloning weights and momentum buffers)...")
    model_A = deepcopy(model).to(device)
    model_B = deepcopy(model).to(device)

    opt_A = torch.optim.SGD(model_A.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0)
    opt_B = torch.optim.SGD(model_B.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0)
    copy_optimizer_state(optimizer, opt_A, device)
    copy_optimizer_state(optimizer, opt_B, device)

    sched_A = WarmupStableDecayScheduler(opt_A, base_lr=base_lr, total_steps=branch_steps, warmup_frac=0.0, decay_frac=0.10)
    sched_B = WarmupStableDecayScheduler(opt_B, base_lr=base_lr, total_steps=branch_steps, warmup_frac=0.0, decay_frac=0.10)

    # Rung 1: Inject single batch perturbation at onset
    if source == "R1":
        print("    [Rung 1] Injecting Gaussian perturbation at sigma=1e-2...")
        with torch.no_grad():
            for p in model_B.parameters():
                noise = torch.randn_like(p) * 0.01 * p.std().clamp(min=1e-4)
                p.add_(noise)

    # 6. Train Branches
    iter_A = iter(loader_A)
    iter_B = iter(loader_B)

    # Save checkpoints every 1% of branch steps for Matched Design
    ckpt_interval = max(1, int(0.01 * branch_steps))
    log_interval = max(500, branch_steps // 10)
    trajectory_ckpts = []
    start_branch_time = time.time()

    print(f"--> Training branches for {branch_steps} steps (Checkpointing every {ckpt_interval} steps)...")
    for b_step in range(branch_steps):
        # Branch A step
        try:
            xA, yA = next(iter_A)
        except StopIteration:
            iter_A = iter(loader_A)
            xA, yA = next(iter_A)
        xA, yA = xA.to(device), yA.to(device)
        opt_A.zero_grad()
        lossA = criterion(model_A(xA), yA)
        lossA.backward()
        opt_A.step()
        sched_A.step()

        # Branch B step
        try:
            xB, yB = next(iter_B)
        except StopIteration:
            iter_B = iter(loader_B)
            xB, yB = next(iter_B)
        xB, yB = xB.to(device), yB.to(device)
        opt_B.zero_grad()
        lossB = criterion(model_B(xB), yB)
        lossB.backward()
        opt_B.step()
        sched_B.step()

        # Checkpoint trajectory
        if save_checkpoints and ((b_step + 1) % ckpt_interval == 0 or b_step == branch_steps - 1):
            dists = compute_parameter_distances(model_A, model_B, model_origin)
            trajectory_ckpts.append({
                "branch_step": b_step + 1,
                "pct_branch": (b_step + 1) / max(branch_steps, 1),
                "d_pair": dists["d_pair"],
                "d_origin": dists.get("d_origin", 0.0)
            })

        if (b_step + 1) % log_interval == 0 or (b_step + 1) == branch_steps:
            elapsed_b = time.time() - start_branch_time
            print(f"    Branch Step {b_step + 1:5d}/{branch_steps} | Loss A: {lossA.item():.4f} | Loss B: {lossB.item():.4f} | Elapsed: {elapsed_b:.1f}s")

    # 7. Endpoint Measurements
    print("\n--> Evaluating endpoints and merge barrier metrics...")
    loss_A, acc_A = evaluate_model(model_A, test_loader, device)
    loss_B, acc_B = evaluate_model(model_B, test_loader, device)
    print(f"    Branch A Test Acc: {acc_A * 100.0:.2f}% | Branch B Test Acc: {acc_B * 100.0:.2f}%")

    # Parameter distances
    dist_metrics = compute_parameter_distances(model_A, model_B, model_origin)
    print(f"    D_pair: {dist_metrics['d_pair']:.5f} (Body: {dist_metrics['d_body']:.5f}, Head: {dist_metrics['d_head']:.5f})")

    # Interpolated barriers (Sharma et al. 11-point path)
    barrier_results = compute_interpolated_barriers(model_A, model_B, test_loader, device, num_points=11)
    print(f"    Loss Barrier: {barrier_results['max_loss_barrier']:.4f} | Accuracy Dip: {barrier_results['max_acc_dip'] * 100.0:.2f}%")
    print(f"    Merged Accuracy: {barrier_results['merged_acc'] * 100.0:.2f}%")

    # Directional curvature
    curv_metrics = compute_directional_curvature(model_A, model_B, test_loader, device)
    print(f"    Curvature: Total={curv_metrics['curvature_total']:.2f} | Body={curv_metrics['curvature_body']:.2f} | Head={curv_metrics['curvature_head']:.2f}")

    # Angular CKA on last-block representations
    feat_A, _ = extract_penultimate_features(model_A, probe_loader, device)
    feat_B, _ = extract_penultimate_features(model_B, probe_loader, device)
    ang_cka = compute_angular_cka(feat_A, feat_B)
    print(f"    Angular CKA (Last block): {ang_cka:.4f}")

    # Functional decomposition ladder: Probe Gap & Few-Shot Head Gain
    model_merged = deepcopy(model_A).to(device)
    from analysis.scientific_metrics import interpolate_state_dicts
    model_merged.load_state_dict(interpolate_state_dicts(model_A.state_dict(), model_B.state_dict(), 0.5))

    loader_train_balanced = DataLoader(train_set, batch_size=128, shuffle=False)
    probe_acc, probe_gap = evaluate_linear_probe_gap(model_merged, loader_train_balanced, test_loader, device, epochs=5)
    few_shot_acc, few_shot_gain = evaluate_few_shot_head_gain(model_merged, few_shot_loader, test_loader, device, epochs=5)
    
    # Baseline no-branch reference accuracy for CIFAR-10
    no_branch_ref_acc = 0.880
    head_share = compute_head_share(probe_gap, no_branch_ref_acc, barrier_results["merged_acc"])
    print(f"    Linear Probe Acc: {probe_acc * 100.0:.2f}% (Probe Gap: {probe_gap * 100.0:+.2f}%)")
    print(f"    Few-Shot Head Acc: {few_shot_acc * 100.0:.2f}% (Few-Shot Gain: {few_shot_gain * 100.0:+.2f}%)")
    print(f"    Head Share of Gap: {head_share * 100.0:.1f}%")

    # Cumulative partial merges profile (j=0..4)
    partial_merges = evaluate_cumulative_partial_merges(model_A, model_B, test_loader, device)
    print(f"    Cumulative Partial Merges: {partial_merges}")

    # Logit-offset fidelity
    p_A = np.ones(10) / 10.0
    p_B = np.ones(10) / 10.0
    if source == "R3-L":
        from fl_outage.data.partition import compute_group_class_proportions
        if asymmetric_mode == "A_skew_B_iid":
            p_A, _ = compute_group_class_proportions(num_classes=10, s=s)
            p_B = np.ones(10) / 10.0
        elif asymmetric_mode is None:
            p_A, p_B = compute_group_class_proportions(num_classes=10, s=s)
    logit_fidelity = compute_logit_offset_fidelity(model_A, model_B, test_loader, p_A, p_B, device)
    print(f"    Logit-Offset Fidelity (H-B2): Slope={logit_fidelity['slope']:.3f} | R2={logit_fidelity['r2']:.3f}")

    # Optional: save branch endpoint weights for rapid Stage 2 head cures
    if save_models:
        if asymmetric_mode:
            model_name_base = f"model_{source}_{asymmetric_mode}_onset{onset_pct}_seed{seed}"
        elif source == "R3-F" and corruption_level:
            model_name_base = f"model_{source}_{corruption_level}_onset{onset_pct}_seed{seed}"
        else:
            model_name_base = f"model_{source}_s{s}_onset{onset_pct}_seed{seed}"
        path_A = os.path.join(output_dir, f"{model_name_base}_branchA.pt")
        path_B = os.path.join(output_dir, f"{model_name_base}_branchB.pt")
        torch.save(model_A.state_dict(), path_A)
        torch.save(model_B.state_dict(), path_B)
        print(f"    Saved branch models to {path_A} and {path_B}")

    # Compile master result
    record = {
        "config": {
            "total_steps": total_steps,
            "onset_pct": onset_pct,
            "source": source,
            "s": s,
            "asymmetric_mode": asymmetric_mode,
            "corruption_level": corruption_level,
            "seed": seed,
            "device": device_name
        },
        "endpoints": {
            "loss_A": float(loss_A),
            "acc_A": float(acc_A),
            "loss_B": float(loss_B),
            "acc_B": float(acc_B),
            "acc_onset": float(acc_onset)
        },
        "distances": dist_metrics,
        "barriers": barrier_results,
        "curvature": curv_metrics,
        "representation": {
            "angular_cka": ang_cka
        },
        "functional_decomposition": {
            "probe_acc": probe_acc,
            "probe_gap": probe_gap,
            "few_shot_acc": few_shot_acc,
            "few_shot_gain": few_shot_gain,
            "head_share": head_share,
            "partial_merges": partial_merges,
            "logit_offset_fidelity": logit_fidelity
        },
        "trajectory_ckpts": trajectory_ckpts
    }

    # Save to disk
    if asymmetric_mode:
        out_filename = f"result_{source}_{asymmetric_mode}_onset{onset_pct}_seed{seed}.json"
    elif source == "R3-F" and corruption_level:
        out_filename = f"result_{source}_{corruption_level}_onset{onset_pct}_seed{seed}.json"
    else:
        out_filename = f"result_{source}_s{s}_onset{onset_pct}_seed{seed}.json"

    out_path = os.path.join(output_dir, out_filename)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n--> Experiment completed! Saved results to {out_path}\n" + "=" * 80)
    return record


def main():
    parser = argparse.ArgumentParser(description="Science Harness: Onset-Dependent Mergeability Runner")
    parser.add_argument("--total_steps", type=int, default=25000, help="Total step budget (reference = 25,000)")
    parser.add_argument("--onset_pct", type=float, default=2.0, help="Onset percentage in {0.0, 0.5, 2.0, 10.0, 25.0, 50.0}")
    parser.add_argument("--source", type=str, default="R3-L", choices=["R1", "R2a", "R2b", "R3-L", "R3-F"])
    parser.add_argument("--s", type=float, default=0.8, help="Label separation parameter in {0.0, 0.5, 0.8, 0.95, 1.0}")
    parser.add_argument("--asymmetric_mode", type=str, default=None, choices=[None, "A_skew_B_iid", "ratio_70_30"])
    parser.add_argument("--corruption_level", type=str, default=None, choices=[None, "mild", "severe"], help="Feature corruption level for R3-F")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/cache", help="Persistent dataset cache directory")
    parser.add_argument("--output_dir", type=str, default="./runs/science_harness")
    parser.add_argument("--save_models", action="store_true", help="Save final branch model state dicts (.pt)")
    args = parser.parse_args()

    run_science_experiment(
        total_steps=args.total_steps,
        onset_pct=args.onset_pct,
        source=args.source,
        s=args.s,
        asymmetric_mode=args.asymmetric_mode,
        corruption_level=args.corruption_level,
        seed=args.seed,
        device_name=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        save_models=args.save_models
    )


if __name__ == "__main__":
    main()
