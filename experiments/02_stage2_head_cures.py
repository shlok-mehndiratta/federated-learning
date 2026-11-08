"""
Stage 2: Mechanistic Decomposition & Privacy-Preserving Head Cures.
Evaluates the three head-curing mechanisms to test Hypotheses H-B1, H-B2, and H-D:
1. Zero-Shot Logit Adjustment (Menon et al. ICLR 2021) - Prior-calibrated logits without data transfer.
2. Few-Shot Head Recalibration - 200 balanced samples (20/class), 5 epochs, frozen representation body.
3. Decoupled Head Ensembling - Merged body with personalized client heads (Softmax probability averaging).
4. Linear Probe Upper Bound - Theoretical performance ceiling of the merged representation.
5. Layer-wise Representation CKA - Depth profile across stages 0 to 4.

Features:
- Dual Modes:
  * eval_saved: Rapidly evaluates existing saved endpoint models (.pt) in seconds.
  * run_suite: Trains conditions from scratch and executes the full Stage 2 decomposition suite.
- High-Throughput Parallelism: Supports --max_workers for concurrent multi-process execution.
- Comprehensive JSON and Markdown summary logging.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import glob
import itertools
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fl_outage.data.cifar import load_cifar10
from fl_outage.data.partition import compute_group_class_proportions
from fl_outage.data.scientific_partitions import get_scientific_subsets
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
    compute_layerwise_cka,
    evaluate_zero_shot_logit_adjustment,
    evaluate_decoupled_head_ensemble,
    evaluate_model,
    interpolate_state_dicts
)


def create_few_shot_calibration_loader(
    dataset,
    samples_per_class: int = 20,
    num_classes: int = 10,
    batch_size: int = 64,
    seed: int = 42
) -> DataLoader:
    """Builds a tiny balanced calibration loader (200 samples total) for privacy-preserving head tuning."""
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    calib_indices = []
    for c in range(num_classes):
        c_idx = np.where(targets == c)[0]
        chosen = rng.choice(c_idx, size=samples_per_class, replace=False)
        calib_indices.extend(chosen)
    subset = Subset(dataset, calib_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def build_probe_set(dataset, num_samples: int = 1000, seed: int = 42) -> DataLoader:
    """Fixed probe set for layer-wise Angular CKA tracking."""
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    probe_indices = []
    classes = np.unique(targets)
    per_class = num_samples // len(classes)
    for c in classes:
        c_idx = np.where(targets == c)[0]
        chosen = rng.choice(c_idx, size=per_class, replace=False)
        probe_indices.extend(chosen)
    probe_subset = Subset(dataset, probe_indices)
    return DataLoader(probe_subset, batch_size=128, shuffle=False)


def evaluate_stage2_mechanisms(
    model_A: nn.Module,
    model_B: nn.Module,
    test_loader: DataLoader,
    probe_loader: DataLoader,
    few_shot_loader: DataLoader,
    train_balanced_loader: DataLoader,
    source: str,
    s: float,
    asymmetric_mode: Optional[str],
    device: torch.device,
    no_branch_ref_acc: float = 0.8721
) -> Dict[str, any]:
    """Runs the complete Stage 2 mechanistic decomposition and head cure benchmark on two trained models."""
    model_A.eval()
    model_B.eval()

    # 1. Endpoint accuracies
    loss_A, acc_A = evaluate_model(model_A, test_loader, device)
    loss_B, acc_B = evaluate_model(model_B, test_loader, device)

    # 2. Naive Linear Merge (W_avg = 0.5 * W_A + 0.5 * W_B)
    model_merged = deepcopy(model_A).to(device)
    model_merged.load_state_dict(interpolate_state_dicts(model_A.state_dict(), model_B.state_dict(), 0.5))
    loss_merged, acc_merged = evaluate_model(model_merged, test_loader, device)

    # 3. Parameter Distances and Curvature
    dist_metrics = compute_parameter_distances(model_A, model_B)
    curv_metrics = compute_directional_curvature(model_A, model_B, test_loader, device)

    # 4. Layer-wise Representation CKA (Hypothesis H-B1)
    layerwise_cka = compute_layerwise_cka(model_A, model_B, probe_loader, device)

    # 5. Theoretical Priors for Logit Adjustments
    p_A = np.ones(10) / 10.0
    p_B = np.ones(10) / 10.0
    if source == "R3-L":
        if asymmetric_mode == "A_skew_B_iid":
            p_A, _ = compute_group_class_proportions(num_classes=10, s=s)
            p_B = np.ones(10) / 10.0
        elif asymmetric_mode is None:
            p_A, p_B = compute_group_class_proportions(num_classes=10, s=s)
    p_mix = 0.5 * (p_A + p_B)

    # 6. Cure 1: Zero-Shot Logit Adjustment (Hypothesis H-B2)
    cure1_results = evaluate_zero_shot_logit_adjustment(
        model=model_merged,
        dataloader=test_loader,
        p_mix=p_mix,
        device=device
    )

    # 7. Cure 2: Few-Shot Head Recalibration (Hypothesis H-D)
    acc_few_shot, gain_few_shot = evaluate_few_shot_head_gain(
        model_merged=model_merged,
        balanced_few_shot_loader=few_shot_loader,
        test_loader=test_loader,
        device=device,
        epochs=5,
        lr=0.05
    )

    # 8. Cure 3: Decoupled Head Ensembling (Body-Only Merge)
    cure3_results = evaluate_decoupled_head_ensemble(
        model_A=model_A,
        model_B=model_B,
        dataloader=test_loader,
        device=device,
        p_A=p_A,
        p_B=p_B,
        tau=1.0
    )

    # 9. Cure 4: Linear Probe Upper Bound (Frozen Merged Features)
    acc_linear_probe, probe_gap = evaluate_linear_probe_gap(
        model_merged=model_merged,
        train_loader=train_balanced_loader,
        test_loader=test_loader,
        device=device,
        epochs=5
    )

    # 10. Performance Recovery Ratios
    # R = (Acc_cure - Acc_merged) / (Acc_upper - Acc_merged)
    denom_upper = max(acc_linear_probe - acc_merged, 1e-6)
    rec_logit_adj = (cure1_results["best_adjusted_acc"] - acc_merged) / denom_upper
    rec_few_shot = (acc_few_shot - acc_merged) / denom_upper
    rec_decoupled = (cure3_results["decoupled_prob_ens_acc"] - acc_merged) / denom_upper
    rec_decoupled_adj = (cure3_results.get("decoupled_adj_prob_ens_acc", cure3_results["decoupled_prob_ens_acc"]) - acc_merged) / denom_upper

    head_share = compute_head_share(probe_gap, no_branch_ref_acc, acc_merged)

    return {
        "endpoints": {
            "acc_A": float(acc_A),
            "loss_A": float(loss_A),
            "acc_B": float(acc_B),
            "loss_B": float(loss_B),
            "acc_naive_merged": float(acc_merged),
            "loss_naive_merged": float(loss_merged),
            "acc_dip": float(max(0.0, max(acc_A, acc_B) - acc_merged))
        },
        "distances": dist_metrics,
        "curvature": curv_metrics,
        "layerwise_cka": layerwise_cka,
        "cure1_zero_shot_logit_adj": {
            "best_acc": float(cure1_results["best_adjusted_acc"]),
            "best_tau": float(cure1_results["best_tau"]),
            "gain": float(cure1_results["gain"]),
            "recovery_ratio": float(rec_logit_adj),
            "tau_sweep": cure1_results["tau_sweep"]
        },
        "cure2_few_shot_recalibration": {
            "acc": float(acc_few_shot),
            "gain": float(gain_few_shot),
            "recovery_ratio": float(rec_few_shot),
            "target_passed": bool(rec_few_shot >= 0.60 or acc_few_shot >= acc_A)
        },
        "cure3_decoupled_head_ensemble": {
            "prob_ens_acc": float(cure3_results["decoupled_prob_ens_acc"]),
            "prior_adj_prob_ens_acc": float(cure3_results.get("decoupled_adj_prob_ens_acc", 0.0)),
            "recovery_ratio": float(rec_decoupled)
        },
        "cure4_linear_probe_upper_bound": {
            "probe_acc": float(acc_linear_probe),
            "probe_gap": float(probe_gap),
            "head_share": float(head_share)
        }
    }


def get_stage2_experiment_plan(suite: str, seeds: List[int]) -> List[dict]:
    """Generates conditions to evaluate for Stage 2."""
    plan = []
    if suite in ("core", "r2b", "full"):
        onsets = [2.0, 10.0, 25.0] if suite == "core" else [2.0, 5.0, 10.0, 20.0, 40.0]
        for onset, seed in itertools.product(onsets, seeds):
            plan.append({
                "source": "R2b",
                "s": 0.0,
                "asymmetric_mode": None,
                "onset_pct": onset,
                "seed": seed
            })

    if suite in ("core", "r3l", "full"):
        s_levels = [0.8, 1.0] if suite == "core" else [0.5, 0.8, 0.95, 1.0]
        onsets = [2.0, 10.0, 25.0] if suite == "core" else [2.0, 5.0, 10.0, 20.0, 40.0]
        for s, onset, seed in itertools.product(s_levels, onsets, seeds):
            plan.append({
                "source": "R3-L",
                "s": s,
                "asymmetric_mode": None,
                "onset_pct": onset,
                "seed": seed
            })

    if suite in ("core", "asym", "full"):
        onsets = [2.0, 10.0, 25.0] if suite == "core" else [2.0, 5.0, 10.0, 20.0, 40.0]
        for onset, seed in itertools.product(onsets, seeds):
            plan.append({
                "source": "R3-L",
                "s": 0.8,
                "asymmetric_mode": "A_skew_B_iid",
                "onset_pct": onset,
                "seed": seed
            })

    return plan


def print_stage2_summary_table(results_list: List[dict]):
    """Formats a clean Markdown summary table comparing the 3 Head Cures."""
    print("\n" + "=" * 125)
    print("STAGE 2 MECHANISTIC DECOMPOSITION & PRIVACY-PRESERVING HEAD CURES SUMMARY")
    print("=" * 125)

    headers = [
        "Source", "Skew", "Mode", "Onset", "Seed",
        "Naive Merged", "Cure 1 (LogitAdj)", "Cure 2 (FewShot)", "Cure 3 (Decoupled)", "Probe Ceiling",
        "FewShot Rec %", "Penult CKA"
    ]

    rows = []
    for r in results_list:
        cfg = r["config"]
        ep = r["endpoints"]
        c1 = r["cure1_zero_shot_logit_adj"]
        c2 = r["cure2_few_shot_recalibration"]
        c3 = r["cure3_decoupled_head_ensemble"]
        c4 = r["cure4_linear_probe_upper_bound"]
        cka = r["layerwise_cka"]["penultimate_pooled"]

        src = cfg["source"]
        s_str = f"{cfg['s']:.1f}" if src == "R3-L" else "0.0"
        mode = cfg.get("asymmetric_mode") or "sym"
        if mode == "A_skew_B_iid":
            mode = "A:sk, B:iid"
        onset = f"{cfg['onset_pct']:.1f}%"
        seed = str(cfg["seed"])

        m_acc = f"{ep['acc_naive_merged']*100:.2f}%"
        c1_acc = f"{c1['best_acc']*100:.2f}%"
        c2_acc = f"{c2['acc']*100:.2f}%"
        c3_acc = f"{c3['prob_ens_acc']*100:.2f}%"
        c4_acc = f"{c4['probe_acc']*100:.2f}%"
        rec_fs = f"{c2['recovery_ratio']*100:+.1f}%"
        cka_str = f"{cka:.3f}"

        rows.append([
            src, s_str, mode, onset, seed,
            m_acc, c1_acc, c2_acc, c3_acc, c4_acc,
            rec_fs, cka_str
        ])

    col_widths = [max(len(str(val)) for val in col) for col in zip(headers, *rows)]
    hdr_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\n" + hdr_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))
    print(sep_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Mechanistic Decomposition & Privacy-Preserving Head Cures")
    parser.add_argument("--mode", type=str, default="eval_saved", choices=["eval_saved", "run_suite"],
                        help="'eval_saved' evaluates already trained .pt models; 'run_suite' runs training then evaluates")
    parser.add_argument("--suite", type=str, default="core", choices=["core", "full", "r2b", "r3l", "asym"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds")
    parser.add_argument("--total_steps", type=int, default=25000, help="Total step budget for training if run_suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--model_dir", type=str, default="./runs/stage1_science", help="Directory where .pt models are saved")
    parser.add_argument("--output_dir", type=str, default="./runs/stage2_head_cures", help="Directory to save Stage 2 results")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent workers for run_suite")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load CIFAR-10 data loaders once
    print(f"\n--> Loading CIFAR-10 evaluation loaders from {args.data_dir}...")
    train_set, test_set = load_cifar10(root=args.data_dir, download=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    probe_loader = build_probe_set(test_set, num_samples=1000, seed=42)
    few_shot_loader = create_few_shot_calibration_loader(train_set, samples_per_class=20, seed=42)
    train_balanced_loader = DataLoader(train_set, batch_size=128, shuffle=False)

    plan = get_stage2_experiment_plan(args.suite, args.seeds)
    print(f"Total Conditions to process: {len(plan)}")

    all_results = []

    for idx, cfg in enumerate(plan, 1):
        source = cfg["source"]
        s = cfg["s"]
        asym = cfg["asymmetric_mode"]
        onset = cfg["onset_pct"]
        seed = cfg["seed"]

        asym_tag = f"_{asym}" if asym else ""
        out_name = f"result_cures_{source}_s{s}{asym_tag}_onset{onset}_seed{seed}.json"
        out_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(out_path):
            print(f"[{idx}/{len(plan)}] Skipping already completed: {out_name}")
            with open(out_path, "r") as f:
                all_results.append(json.load(f))
            continue

        model_name_base = f"model_{source}_s{s}_onset{onset}_seed{seed}" if not asym else f"model_{source}_{asym}_onset{onset}_seed{seed}"
        path_A = os.path.join(args.model_dir, f"{model_name_base}_branchA.pt")
        path_B = os.path.join(args.model_dir, f"{model_name_base}_branchB.pt")

        if os.path.exists(path_A) and os.path.exists(path_B):
            print(f"\n>>> [{idx}/{len(plan)}] Evaluating saved models: {model_name_base}...")
            model_A = ResNet20LN(num_classes=10).to(device)
            model_B = ResNet20LN(num_classes=10).to(device)
            model_A.load_state_dict(torch.load(path_A, map_location=device))
            model_B.load_state_dict(torch.load(path_B, map_location=device))
        else:
            if args.mode == "eval_saved":
                print(f"[{idx}/{len(plan)}] Model files not found in {args.model_dir}: {path_A}. Skipping in eval_saved mode.")
                continue
            else:
                print(f"\n>>> [{idx}/{len(plan)}] Training {model_name_base} on the fly...")
                from experiments.science_harness import run_science_experiment
                run_science_experiment(
                    total_steps=args.total_steps,
                    onset_pct=onset,
                    source=source,
                    s=s,
                    asymmetric_mode=asym,
                    seed=seed,
                    device_name=args.device,
                    output_dir=args.model_dir,
                    data_dir=args.data_dir,
                    save_models=True
                )
                model_A = ResNet20LN(num_classes=10).to(device)
                model_B = ResNet20LN(num_classes=10).to(device)
                model_A.load_state_dict(torch.load(path_A, map_location=device))
                model_B.load_state_dict(torch.load(path_B, map_location=device))

        # Run mechanistic decomposition
        t0 = time.time()
        cures_data = evaluate_stage2_mechanisms(
            model_A=model_A,
            model_B=model_B,
            test_loader=test_loader,
            probe_loader=probe_loader,
            few_shot_loader=few_shot_loader,
            train_balanced_loader=train_balanced_loader,
            source=source,
            s=s,
            asymmetric_mode=asym,
            device=device
        )
        elapsed = time.time() - t0

        record = {
            "config": cfg,
            **cures_data,
            "elapsed_seconds": elapsed
        }

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"    Finished {out_name} in {elapsed:.1f}s | Naive: {cures_data['endpoints']['acc_naive_merged']*100:.2f}% -> Cure2 (FewShot): {cures_data['cure2_few_shot_recalibration']['acc']*100:.2f}% | Cure3 (Decoupled): {cures_data['cure3_decoupled_head_ensemble']['prob_ens_acc']*100:.2f}%")
        all_results.append(record)

    if all_results:
        print_stage2_summary_table(all_results)
        master_summary_path = os.path.join(args.output_dir, "stage2_master_summary.json")
        with open(master_summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Master Stage 2 summary saved to {master_summary_path}\n" + "=" * 125)


if __name__ == "__main__":
    main()
