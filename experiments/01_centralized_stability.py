"""Experiment 01: Milestone 1 - Centralized SGD Stability Boundary Evaluation.

Implements the centralized spawn-and-perturb protocol on CIFAR-10 with ResNet-20 LayerNorm:
1. Trains a base model with SGD (lr=0.1, momentum=0.9, triangular schedule).
2. At specified onset step t in {0%, 0.5%, 2%, 10%, 50%}, forks into Copy 1 and Copy 2.
3. Applies a controlled batch-gradient perturbation to Copy 2.
4. From step t to end, both copies train with identical deterministic batches.
5. Measures Sharma et al. 11-point loss barrier, parameter distance D_pair, and test accuracy.
"""

import argparse
from copy import deepcopy
import json
import math
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_outage.data.cifar import load_cifar10, create_deterministic_loader
from fl_outage.models.resnet import ResNet20LN
from fl_outage.metrics.barrier import compute_linear_barrier, evaluate_loss_and_acc
from fl_outage.metrics.distances import compute_normalized_distances

try:
    import wandb
except ImportError:
    wandb = None


def get_lr_triangle(step: int, total_steps: int, base_lr: float = 0.1, warmup_ratio: float = 0.02) -> float:
    """Computes learning rate with linear warmup and linear decay (triangular schedule)."""
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    decay_steps = total_steps - warmup_steps
    progress = float(step - warmup_steps) / float(max(1, decay_steps))
    return max(0.0, base_lr * (1.0 - progress))


def apply_batch_perturbation(
    model: nn.Module,
    sample_loader: DataLoader,
    device: torch.device,
    scale: float = 0.01
) -> Dict[str, float]:
    """Applies a single-batch gradient perturbation to model weights:
        W <- W + scale * (grad / ||grad||)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get a single batch
    x, y = next(iter(sample_loader))
    x, y = x.to(device), y.to(device)
    
    model.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    # Collect gradients and compute total norm
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad.data ** 2).item()
    total_norm = math.sqrt(total_norm_sq)

    # Apply scaled perturbation
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None and total_norm > 0:
                p.data.add_(p.grad.data, alpha=(scale / total_norm))
    model.zero_grad()

    return {"perturb_scale": scale, "grad_norm": total_norm}


def run_v1_fork(
    onset_step: int,
    total_steps: int,
    seed: int,
    device: torch.device,
    train_set,
    test_set,
    batch_size: int = 128,
    base_lr: float = 0.1,
    momentum: float = 0.9,
    perturb_scale: float = 0.01,
    num_eval_points: int = 11,
    use_wandb: bool = False
) -> Dict[str, any]:
    """Runs a single spawn-and-perturb fork from onset_step to total_steps."""
    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize shared model
    model = ResNet20LN(num_classes=10, width_base=16).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed)
    test_loader = create_deterministic_loader(test_set, batch_size=256, shuffle=False)
    data_iter = iter(train_loader)

    # Phase 1: Train base model from step 0 to onset_step
    print(f"\n[Seed {seed}] Training base model to onset step {onset_step}/{total_steps}...")
    for step in range(onset_step):
        try:
            x, y = next(data_iter)
        except StopIteration:
            train_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed + step)
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        lr = get_lr_triangle(step, total_steps, base_lr=base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Phase 2: Fork into Copy 1 and Copy 2
    model_copy1 = deepcopy(model)
    model_copy2 = deepcopy(model)

    # Apply perturbation to Copy 2
    perturb_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed + 9999)
    apply_batch_perturbation(model_copy2, perturb_loader, device=device, scale=perturb_scale)

    opt1 = torch.optim.SGD(model_copy1.parameters(), lr=base_lr, momentum=momentum)
    opt2 = torch.optim.SGD(model_copy2.parameters(), lr=base_lr, momentum=momentum)

    # Phase 3: Train both copies with identical batches from onset_step to total_steps
    remaining_steps = total_steps - onset_step
    print(f"[Seed {seed}] Training both copies for remaining {remaining_steps} steps with identical data...")

    for step in range(onset_step, total_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            train_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed + step)
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        lr = get_lr_triangle(step, total_steps, base_lr=base_lr)
        for g in opt1.param_groups:
            g["lr"] = lr
        for g in opt2.param_groups:
            g["lr"] = lr

        x, y = x.to(device), y.to(device)

        # Copy 1 step
        opt1.zero_grad()
        out1 = model_copy1(x)
        loss1 = criterion(out1, y)
        loss1.backward()
        opt1.step()

        # Copy 2 step (identical batch x, y)
        opt2.zero_grad()
        out2 = model_copy2(x)
        loss2 = criterion(out2, y)
        loss2.backward()
        opt2.step()

    # Phase 4: Measurement
    print(f"[Seed {seed}] Evaluating Sharma et al. 11-point barrier on test set...")
    barrier_res = compute_linear_barrier(model_copy1, model_copy2, test_loader, num_points=num_eval_points, device=device)
    dist_res = compute_normalized_distances(model_copy1, model_copy2, model_anchor=model)
    _, acc1 = evaluate_loss_and_acc(model_copy1, test_loader, device=device)
    _, acc2 = evaluate_loss_and_acc(model_copy2, test_loader, device=device)

    result = {
        "onset_step": onset_step,
        "onset_pct": float(onset_step) / float(total_steps) * 100.0,
        "seed": seed,
        "loss_barrier": barrier_res["barrier"],
        "acc_dip": barrier_res["max_acc_dip"],
        "d_pair": dist_res["d_pair"],
        "d_origin": dist_res["d_origin"],
        "acc_copy1": acc1,
        "acc_copy2": acc2,
        "merged_acc": barrier_res["merged_acc"],
    }

    print(f"--> Onset {result['onset_pct']:.1f}% | Loss Barrier: {result['loss_barrier']:.4f} | D_pair: {result['d_pair']:.4f} | Merged Acc: {result['merged_acc']*100:.2f}%")

    if use_wandb and wandb is not None:
        wandb.log({
            "v1/onset_pct": result["onset_pct"],
            "v1/loss_barrier": result["loss_barrier"],
            "v1/d_pair": result["d_pair"],
            "v1/merged_acc": result["merged_acc"],
            "v1/seed": seed
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Milestone 1: Centralized SGD Stability Boundary Evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default 50 = 19,531 steps)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--perturb_scale", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./runs/centralized_stability")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fl-centralized-stability")
    parser.add_argument("--toy_run", action="store_true", help="Quick smoke test on small steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running Centralized Stability Evaluation on device: {device}")

    # Load CIFAR-10
    train_set, test_set = load_cifar10(root=args.data_root, download=True)
    steps_per_epoch = len(train_set) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    if args.toy_run:
        print("--- TOY RUN MODE (Smoke test) ---")
        total_steps = 20
        onset_steps = [0, 5, 10]
        args.seeds = [1]
    else:
        # Standard onset percentages from Kwok & Altintas: {0%, 0.5%, 2%, 10%, 50%}
        onset_pcts = [0.0, 0.005, 0.02, 0.10, 0.50]
        onset_steps = [int(p * total_steps) for p in onset_pcts]

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    out_file = os.path.join(args.output_dir, "centralized_results.json")
    # Support backward compatibility with existing v1_results.json
    legacy_file = os.path.join(args.output_dir, "v1_results.json")
    if not os.path.exists(out_file) and os.path.exists(legacy_file):
        out_file = legacy_file
    all_results = []
    completed_keys = set()
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                all_results = json.load(f)
            for r in all_results:
                completed_keys.add((r["onset_step"], r["seed"]))
            print(f"Loaded {len(all_results)} existing results from {out_file}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not read existing results ({e}). Starting fresh.")

    for onset in onset_steps:
        for seed in args.seeds:
            if (onset, seed) in completed_keys:
                print(f"--> Skipping Onset step {onset} (Seed {seed}) - already completed.")
                continue

            res = run_v1_fork(
                onset_step=onset,
                total_steps=total_steps,
                seed=seed,
                device=device,
                train_set=train_set,
                test_set=test_set,
                batch_size=args.batch_size,
                base_lr=args.lr,
                momentum=args.momentum,
                perturb_scale=args.perturb_scale,
                use_wandb=args.use_wandb
            )
            all_results.append(res)
            completed_keys.add((onset, seed))

            # Incremental save so no progress is ever lost
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll V1 runs complete! Results saved to {out_file}")


if __name__ == "__main__":
    main()
