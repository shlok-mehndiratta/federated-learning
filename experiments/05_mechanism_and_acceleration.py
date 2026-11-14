"""Experiment 05: Milestone 4 - Structural Mechanisms, Representational Analysis (CKA), and Recovery Acceleration.

Executes the 4 diagnostic probes to explain and resolve the post-merge barrier:
1. Probe 1: Layer-Wise Barrier Decomposition across Init Conv, Stages 1-3, and Linear Head.
2. Probe 2: Stage-by-Stage Linear Centered Kernel Alignment (CKA) feature similarity.
3. Probe 3: Per-class accuracy, 10x10 confusion attribution, and predictive entropy.
4. Probe 4: Post-merge recovery acceleration:
   - Git Re-Basin Hungarian permutation alignment (weight matching)
   - Post-merge client momentum reset vs retained momentum

Resource Guarantees:
- Peak VRAM < 300 MB via sequential client execution.
- Fast smoke test support via --quick_test.
- Incremental checkpoint saving to mechanism_results.json with auto-resume.
"""

import argparse
from copy import deepcopy
import json
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from fl_outage.data.cifar import load_cifar10, create_deterministic_loader
from fl_outage.data.partition import partition_dataset
from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.server import FederatedServer
from fl_outage.fl.outage import OutageSimulator
from fl_outage.fl.merge import merge_models
from fl_outage.fl.recovery import run_recovery_evaluation
from fl_outage.fl.rebasin import match_weights_resnet20
from fl_outage.metrics.barrier import compute_linear_barrier, evaluate_loss_and_acc
from fl_outage.metrics.distances import compute_normalized_distances
from fl_outage.metrics.curvature import compute_curvature_barrier_estimate
from fl_outage.metrics.cka import compute_stage_cka_summary
from fl_outage.metrics.layer_barrier import compute_layer_wise_barrier_profile


def set_seed(seed: int):
    """Sets deterministic random seed across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_targets(dataset) -> np.ndarray:
    """Safely extracts label targets array from Dataset or Subset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        return np.array(dataset.dataset.targets)[dataset.indices]
    else:
        return np.array([y for _, y in dataset])


@torch.no_grad()
def evaluate_confusion_and_entropy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10
) -> Dict[str, any]:
    """Computes full confusion matrix, per-class recall, cohort accuracy, and entropy."""
    model.eval()
    model.to(device)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    entropies: List[float] = []
    total = 0
    correct = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        # Prediction entropy: -sum p log p
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        entropies.extend(entropy.cpu().numpy().tolist())

        preds = logits.argmax(dim=1)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1
        correct += (preds == y).sum().item()
        total += x.size(0)

    # Per-class recall (accuracy per class)
    class_totals = np.maximum(confusion.sum(axis=1), 1)
    per_class_recall = (np.diag(confusion) / class_totals).tolist()

    half = num_classes // 2
    acc_cohort_A = float(np.mean(per_class_recall[:half]))
    acc_cohort_B = float(np.mean(per_class_recall[half:]))

    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
        "acc_cohort_A": acc_cohort_A,
        "acc_cohort_B": acc_cohort_B,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "total_accuracy": correct / max(total, 1)
    }


def precompute_global_checkpoints(
    seed: int,
    train_set,
    targets: np.ndarray,
    eval_loader,
    checkpoint_rounds: List[int],
    total_baseline_rounds: int,
    num_clients: int,
    width_base: int,
    lr: float,
    local_steps: int,
    device: torch.device,
    cache_dir: str
) -> Dict[int, str]:
    """Caches unbroken global baseline checkpoints for a given seed."""
    os.makedirs(cache_dir, exist_ok=True)
    paths: Dict[int, str] = {}
    all_exist = True
    for k in checkpoint_rounds:
        p = os.path.join(cache_dir, f"global_model_seed{seed}_round_{k}.pt")
        paths[k] = p
        if not os.path.exists(p):
            all_exist = False

    history_path = os.path.join(cache_dir, f"baseline_history_seed{seed}.json")
    if all_exist and os.path.exists(history_path):
        print(f"--> [Cache Hit] Found cached global checkpoints for Seed {seed}.", flush=True)
        return paths

    print(f"--> [Pre-training] Training unbroken global server model for Seed {seed}...", flush=True)
    set_seed(seed)
    init_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)

    partitions = partition_dataset(
        targets=targets,
        num_classes=10,
        s=0.0,
        alpha=1.0,
        num_clients_per_group=num_clients // 2,
        seed=seed
    )

    server = FederatedServer(
        model=init_model,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        lr=lr,
        local_steps_per_round=local_steps,
        device=device,
        checkpoint_dir=cache_dir
    )

    baseline_history: Dict[int, float] = {}

    if 0 in checkpoint_rounds:
        _, acc0 = evaluate_loss_and_acc(server.model, eval_loader, device=device)
        baseline_history[0] = acc0
        torch.save(server.model.state_dict(), paths[0])

    for r in range(1, total_baseline_rounds + 1):
        server.run_rounds(num_rounds=1, start_step=r * local_steps, seed=seed + r * 100)
        _, round_acc = evaluate_loss_and_acc(server.model, eval_loader, device=device)
        baseline_history[r] = round_acc
        if r in checkpoint_rounds:
            torch.save(server.model.state_dict(), paths[r])

    with open(history_path, "w") as f:
        json.dump(baseline_history, f, indent=2)

    return paths


def run_mechanism_diagnostic(
    condition_name: str,
    s: float,
    k: int,
    isolation: str,
    L: int,
    seed: int,
    base_model: nn.Module,
    train_set,
    partitions: Dict[int, List[int]],
    eval_loader: DataLoader,
    target_baseline_acc: float,
    device: torch.device,
    local_steps: int = 50,
    lr: float = 0.1,
    max_recovery_rounds: int = 20,
    quick_test: bool = False
) -> Dict[str, any]:
    """Runs all 4 diagnostic probes on a single experimental condition."""
    print(f"\n{'='*70}\n[Probe Suite] Condition: {condition_name} | Seed {seed} | s={s} | k={k} | Iso={isolation}\n{'='*70}", flush=True)

    group_A = list(range(len(partitions) // 2))
    group_B = list(range(len(partitions) // 2, len(partitions)))

    # 1. Execute Outage Simulation
    sim = OutageSimulator(
        base_model=base_model,
        dataset=train_set,
        partitions=partitions,
        group_A_cids=group_A,
        group_B_cids=group_B,
        lr=lr,
        local_steps_per_round=local_steps,
        device=device
    )
    mA, mB = sim.run_outage(
        num_rounds=L,
        isolation_level=isolation,
        sparse_freq_R=5,
        seed=seed + k * 10,
        verbose=False
    )

    # 2. Standard FedAvg Midpoint Merge
    m_merged = merge_models(mA, mB)
    _, acc_A = evaluate_loss_and_acc(mA, eval_loader, device=device)
    _, acc_B = evaluate_loss_and_acc(mB, eval_loader, device=device)
    _, acc_merged = evaluate_loss_and_acc(m_merged, eval_loader, device=device)

    # Standard geometric barrier and distance
    barrier_res = compute_linear_barrier(mA, mB, eval_loader, num_points=11 if not quick_test else 3, device=device)
    dist_res = compute_normalized_distances(mA, mB, model_anchor=base_model)
    curvature_est = compute_curvature_barrier_estimate(mA, mB, eval_loader, device=device)

    # Probe 1: Layer-Wise Barrier Decomposition across conv stages and linear head
    print("--> Running Probe 1: Layer-Wise Barrier Decomposition...", flush=True)
    layer_barrier_res = compute_layer_wise_barrier_profile(
        mA, mB, eval_loader,
        num_points=11 if not quick_test else 3,
        device=device
    )

    # Probe 2: Representational Similarity Analysis using Linear CKA
    print("--> Running Probe 2: Linear CKA Representational Alignment...", flush=True)
    cka_res = compute_stage_cka_summary(
        mA, mB, eval_loader,
        device=device,
        max_samples=1000 if not quick_test else 100
    )

    # Probe 3: Per-Class Accuracy and Confusion Matrix Attribution
    print("--> Running Probe 3: Class Confusion Attribution & Entropy...", flush=True)
    conf_A = evaluate_confusion_and_entropy(mA, eval_loader, device)
    conf_B = evaluate_confusion_and_entropy(mB, eval_loader, device)
    conf_merged = evaluate_confusion_and_entropy(m_merged, eval_loader, device)

    # Probe 4: Post-Merge Recovery Acceleration
    print("--> Running Probe 4A: Baseline Standard FedAvg Recovery...", flush=True)
    rec_standard = run_recovery_evaluation(
        merged_model=m_merged,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        baseline_target_acc=target_baseline_acc,
        max_recovery_rounds=max_recovery_rounds if not quick_test else 1,
        lr=lr,
        local_steps_per_round=local_steps,
        momentum=0.0,
        reset_momentum=False,
        device=device,
        seed=seed
    )

    print("--> Running Probe 4B: Git Re-Basin Permutation Alignment Merge...", flush=True)
    perm_B, m_rebasin, rebasin_info = match_weights_resnet20(mA, mB, max_iters=3 if not quick_test else 1)
    _, acc_rebasin = evaluate_loss_and_acc(m_rebasin, eval_loader, device=device)

    rec_rebasin = run_recovery_evaluation(
        merged_model=m_rebasin,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        baseline_target_acc=target_baseline_acc,
        max_recovery_rounds=max_recovery_rounds if not quick_test else 1,
        lr=lr,
        local_steps_per_round=local_steps,
        momentum=0.0,
        reset_momentum=False,
        device=device,
        seed=seed
    )

    print("--> Running Probe 4C: Post-Merge Client Momentum Reset...", flush=True)
    # Compare recovery with momentum=0.9: retained vs reset
    m_merged_mom1 = deepcopy(m_merged)
    m_merged_mom2 = deepcopy(m_merged)

    rec_momentum_retained = run_recovery_evaluation(
        merged_model=m_merged_mom1,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        baseline_target_acc=target_baseline_acc,
        max_recovery_rounds=max_recovery_rounds if not quick_test else 1,
        lr=lr,
        local_steps_per_round=local_steps,
        momentum=0.9,
        reset_momentum=False,
        device=device,
        seed=seed
    )

    rec_momentum_reset = run_recovery_evaluation(
        merged_model=m_merged_mom2,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        baseline_target_acc=target_baseline_acc,
        max_recovery_rounds=max_recovery_rounds if not quick_test else 1,
        lr=lr,
        local_steps_per_round=local_steps,
        momentum=0.9,
        reset_momentum=True,
        device=device,
        seed=seed
    )

    return {
        "condition_name": condition_name,
        "seed": seed,
        "s": s,
        "k": k,
        "isolation": isolation,
        "L": L,
        "target_baseline_acc": target_baseline_acc,
        "endpoints": {
            "acc_A": acc_A,
            "acc_B": acc_B,
            "acc_merged_standard": acc_merged,
            "acc_merged_rebasin": acc_rebasin,
            "rebasin_acc_gain": acc_rebasin - acc_merged
        },
        "barrier": barrier_res,
        "distances": dist_res,
        "curvature": curvature_est,
        "probe1_layer_barrier": {
            "stage_acc_dips": layer_barrier_res["stage_acc_dips"],
            "isolated_merge_impact": layer_barrier_res["isolated_merge_impact"]
        },
        "probe2_cka": cka_res,
        "probe3_attribution": {
            "conf_A": {k: v for k, v in conf_A.items() if k != "confusion_matrix"},
            "conf_B": {k: v for k, v in conf_B.items() if k != "confusion_matrix"},
            "conf_merged": conf_merged
        },
        "probe4_acceleration": {
            "standard_recovery": {
                "rtb": rec_standard["rtb"],
                "final_acc": rec_standard["final_acc"],
                "recovered": rec_standard["recovered"],
                "history": rec_standard["history"]
            },
            "rebasin_recovery": {
                "rtb": rec_rebasin["rtb"],
                "final_acc": rec_rebasin["final_acc"],
                "recovered": rec_rebasin["recovered"],
                "history": rec_rebasin["history"]
            },
            "momentum_retained_recovery": {
                "rtb": rec_momentum_retained["rtb"],
                "final_acc": rec_momentum_retained["final_acc"],
                "recovered": rec_momentum_retained["recovered"],
                "history": rec_momentum_retained["history"]
            },
            "momentum_reset_recovery": {
                "rtb": rec_momentum_reset["rtb"],
                "final_acc": rec_momentum_reset["final_acc"],
                "recovered": rec_momentum_reset["recovered"],
                "history": rec_momentum_reset["history"]
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: Diagnostic Mechanisms & Acceleration Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds (default: 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--data_dir", type=str, default="./data/cache", help="Path to CIFAR-10 data cache")
    parser.add_argument("--output_dir", type=str, default="./runs/mechanism_sweep", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./runs/factorial_sweep/global_checkpoints", help="Global model cache")
    parser.add_argument("--quick_test", action="store_true", help="Run 30-second fast CPU smoke test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"=== MILESTONE 4: MECHANISMS & ACCELERATION ===")
    print(f"Device: {device} | Seeds: {args.seeds} | Quick Test: {args.quick_test}")
    print(f"Output Directory: {args.output_dir}")

    # Load CIFAR-10
    train_set, test_set = load_cifar10(root=args.data_dir)
    if args.quick_test:
        train_set = Subset(train_set, list(range(100)))
        test_eval_set = Subset(test_set, list(range(50)))
        eval_loader = create_deterministic_loader(test_eval_set, batch_size=25, shuffle=False)
    else:
        eval_loader = create_deterministic_loader(test_set, batch_size=128, shuffle=False)
    targets = get_dataset_targets(train_set)

    # Diagnostic conditions to map the mechanistic landscape:
    conditions = [
        {"name": "Failing_Early_Outage", "s": 1.0, "k": 0, "iso": "I0", "L": 20},
        {"name": "Onset_Protected", "s": 1.0, "k": 2, "iso": "I0", "L": 20},
        {"name": "Coupling_Protected", "s": 1.0, "k": 0, "iso": "I2", "L": 20},
        {"name": "Partial_Skew", "s": 0.8, "k": 0, "iso": "I0", "L": 20},
        {"name": "IID_Control", "s": 0.0, "k": 0, "iso": "I0", "L": 20},
    ]
    if args.quick_test:
        conditions = conditions[:1]

    results_file = os.path.join(args.output_dir, "mechanism_results.json")
    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_results = json.load(f)
            print(f"--> [Auto-Resume] Loaded {len(all_results)} existing condition results.")
        except Exception:
            all_results = []

    completed_keys = {(r["seed"], r["condition_name"]) for r in all_results}

    checkpoint_rounds = [0, 2]
    num_clients = 10
    width_base = 16
    local_steps = 50 if not args.quick_test else 2
    max_recovery_rounds = 20 if not args.quick_test else 1

    for seed in args.seeds:
        # Check or precompute global model checkpoints
        paths = precompute_global_checkpoints(
            seed=seed,
            train_set=train_set,
            targets=targets,
            eval_loader=eval_loader,
            checkpoint_rounds=checkpoint_rounds,
            total_baseline_rounds=25,
            num_clients=num_clients,
            width_base=width_base,
            lr=0.1,
            local_steps=local_steps,
            device=device,
            cache_dir=args.checkpoint_dir
        )

        # Baseline history
        history_path = os.path.join(args.checkpoint_dir, f"baseline_history_seed{seed}.json")
        baseline_history = {}
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                baseline_history = {int(k): v for k, v in json.load(f).items()}

        for cond in conditions:
            key = (seed, cond["name"])
            if key in completed_keys:
                print(f"--> [Skip] Already completed: Seed {seed} | {cond['name']}")
                continue

            s = cond["s"]
            k = cond["k"]
            iso = cond["iso"]
            L = cond["L"] if not args.quick_test else 1

            # Base model at round k
            base_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)
            base_model.load_state_dict(torch.load(paths[k], map_location=device, weights_only=True))

            # Partition dataset
            partitions = partition_dataset(
                targets=targets,
                num_classes=10,
                s=s,
                alpha=1.0,
                num_clients_per_group=num_clients // 2,
                seed=seed
            )

            target_acc = baseline_history.get(k + L, 0.40)

            res = run_mechanism_diagnostic(
                condition_name=cond["name"],
                s=s,
                k=k,
                isolation=iso,
                L=L,
                seed=seed,
                base_model=base_model,
                train_set=train_set,
                partitions=partitions,
                eval_loader=eval_loader,
                target_baseline_acc=target_acc,
                device=device,
                local_steps=local_steps,
                lr=0.1,
                max_recovery_rounds=max_recovery_rounds,
                quick_test=args.quick_test
            )

            all_results.append(res)
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"MILESTONE 4 DIAGNOSTIC RUN COMPLETED! Saved to {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
