"""
Stage 3: Feature Skew Control & Falsification Runner (Hypothesis H-E).

Evaluates covariate / feature distribution shift (CIFAR-10-C synthetic corruptions):
- Branch A receives clean CIFAR-10 disjoint half.
- Branch B receives corrupted CIFAR-10 disjoint half (mild / severe).
- Evaluates across onsets k in {2.0%, 10.0%, 25.0%} and random seeds.

The Critical Falsification Test:
Under label skew (R3-L), representation bodies align while heads drift (D_head/D_body > 4x),
and Few-Shot Head Recalibration recovers 40-54% of the probe ceiling gap.
Under feature skew (R3-F), early convolutional filters diverge on disparate texture statistics,
driving body divergence (D_body rises, penultimate CKA drops), causing head-only cures to fail.
This establishes the rigorous theoretical boundary conditions of linear head cures.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fl_outage.data.cifar import load_cifar10
from fl_outage.models.resnet import ResNet20LN
from analysis.scientific_metrics import (
    compute_parameter_distances,
    compute_directional_curvature,
    extract_stage_representations,
    compute_layerwise_cka,
    evaluate_zero_shot_logit_adjustment,
    evaluate_few_shot_head_gain,
    evaluate_decoupled_head_ensemble,
    evaluate_linear_probe_gap,
    evaluate_model
)


def get_feature_skew_plan(corruption_levels: List[str], onsets: List[float], seeds: List[int]) -> List[dict]:
    """Generates factorial experimental conditions for Rung 3-F (Feature Skew)."""
    plan = []
    for level, onset, seed in itertools.product(corruption_levels, onsets, seeds):
        plan.append({
            "source": "R3-F",
            "corruption_level": level,
            "s": 0.0,
            "asymmetric_mode": None,
            "onset_pct": onset,
            "seed": seed
        })
    return plan


def run_single_feature_skew_training(cfg: dict, args) -> Tuple[str, float, int, str]:
    """Executes science_harness training for a single feature skew condition."""
    level = cfg["corruption_level"]
    onset = cfg["onset_pct"]
    seed = cfg["seed"]

    out_filename = f"result_R3-F_{level}_onset{onset}_seed{seed}.json"
    log_name = f"log_R3-F_{level}_onset{onset}_seed{seed}.log"
    log_path = os.path.join(args.output_dir, log_name)

    cmd = [
        sys.executable, "experiments/science_harness.py",
        "--total_steps", str(args.total_steps),
        "--onset_pct", str(onset),
        "--source", "R3-F",
        "--corruption_level", level,
        "--s", "0.0",
        "--seed", str(seed),
        "--device", args.device,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--save_models"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    t0 = time.time()
    with open(log_path, "w") as log_file:
        res = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    return out_filename, elapsed, res.returncode, log_path


def evaluate_feature_skew_cures(
    cfg: dict,
    model_dir: str,
    output_dir: str,
    data_dir: str = "./data/cache",
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, any]:
    """Evaluates mechanistic decomposition and head cures on trained R3-F feature skew models."""
    device = torch.device(device_name)
    level = cfg["corruption_level"]
    onset = cfg["onset_pct"]
    seed = cfg["seed"]

    # Model checkpoint paths
    path_A = os.path.join(model_dir, f"model_R3-F_s0.0_onset{onset}_seed{seed}_branchA.pt")
    path_B = os.path.join(model_dir, f"model_R3-F_s0.0_onset{onset}_seed{seed}_branchB.pt")

    # If named with corruption_level
    if not os.path.exists(path_A):
        alt_path_A = os.path.join(model_dir, f"model_R3-F_{level}_onset{onset}_seed{seed}_branchA.pt")
        alt_path_B = os.path.join(model_dir, f"model_R3-F_{level}_onset{onset}_seed{seed}_branchB.pt")
        if os.path.exists(alt_path_A):
            path_A, path_B = alt_path_A, alt_path_B

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        raise FileNotFoundError(f"Model files not found for R3-F {level} onset {onset} seed {seed} in {model_dir}")

    # Load CIFAR-10 data
    train_set, test_set = load_cifar10(root=data_dir, download=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Balanced calibration sets (200 samples)
    few_shot_indices = []
    rng_fs = np.random.RandomState(seed)
    targets_train = np.array(train_set.targets)
    for c in range(10):
        c_idx = np.where(targets_train == c)[0]
        few_shot_indices.extend(rng_fs.choice(c_idx, size=20, replace=False))
    few_shot_loader = DataLoader(Subset(train_set, few_shot_indices), batch_size=64, shuffle=True)
    probe_train_loader = DataLoader(Subset(train_set, few_shot_indices), batch_size=64, shuffle=False)

    # 1. Instantiate and load models
    model_A = ResNet20LN(num_classes=10).to(device)
    model_B = ResNet20LN(num_classes=10).to(device)
    model_A.load_state_dict(torch.load(path_A, map_location=device, weights_only=True))
    model_B.load_state_dict(torch.load(path_B, map_location=device, weights_only=True))

    criterion = torch.nn.CrossEntropyLoss()
    loss_A, acc_A = evaluate_model(model_A, test_loader, criterion, device)
    loss_B, acc_B = evaluate_model(model_B, test_loader, criterion, device)

    # 2. Naive Midpoint Merged Model
    model_merged = deepcopy(model_A)
    merged_sd = model_merged.state_dict()
    sd_A = model_A.state_dict()
    sd_B = model_B.state_dict()
    for k in merged_sd.keys():
        if merged_sd[k].dtype.is_floating_point:
            merged_sd[k] = 0.5 * (sd_A[k] + sd_B[k])
    model_merged.load_state_dict(merged_sd)
    loss_merged, acc_merged = evaluate_model(model_merged, test_loader, criterion, device)
    baseline_acc = 0.5 * (acc_A + acc_B)
    acc_dip = max(0.0, baseline_acc - acc_merged)

    # 3. Parameter Distances & Directional Curvature
    dist_metrics = compute_parameter_distances(model_A, model_B)
    curv_metrics = compute_directional_curvature(model_A, model_B, test_loader, criterion, device)

    # 4. Layerwise Angular CKA
    probe_indices = []
    rng_p = np.random.RandomState(42)
    targets_test = np.array(test_set.targets)
    for c in range(10):
        c_idx = np.where(targets_test == c)[0]
        probe_indices.extend(rng_p.choice(c_idx, size=100, replace=False))
    probe_loader = DataLoader(Subset(test_set, probe_indices), batch_size=128, shuffle=False)

    reps_A = extract_stage_representations(model_A, probe_loader, device)
    reps_B = extract_stage_representations(model_B, probe_loader, device)
    layerwise_cka = compute_layerwise_cka(reps_A, reps_B)

    # 5. Cure 1: Zero-Shot Logit Adjustment
    prior_ratios = torch.ones(10, dtype=torch.float32, device=device)
    cure1_res = evaluate_zero_shot_logit_adjustment(
        model_merged, test_loader, prior_ratios, device, tau_values=[0.25, 0.5, 1.0, 1.5, 2.0]
    )

    # 6. Cure 2: Few-Shot Head Recalibration (200 balanced samples, 5 epochs, frozen body)
    fs_acc, fs_gain = evaluate_few_shot_head_gain(
        model_merged, few_shot_loader, test_loader, device, epochs=5, lr=0.05
    )

    # 7. Cure 3: Decoupled Head Ensemble
    prior_A = torch.ones(10, device=device) / 10.0
    prior_B = torch.ones(10, device=device) / 10.0
    cure3_res = evaluate_decoupled_head_ensemble(
        model_A, model_B, model_merged, test_loader, prior_A, prior_B, device
    )

    # 8. Cure 4: Linear Probe Upper Bound (Feature Trunk Quality)
    probe_res = evaluate_linear_probe_gap(
        model_merged, train_loader=probe_train_loader, test_loader=test_loader,
        device=device, epochs=10, lr=0.1
    )

    probe_acc = probe_res["probe_acc"]
    probe_gap = probe_res["probe_gap"]
    denom = max(probe_acc - acc_merged, 0.005)
    fs_rec_ratio = fs_gain / denom

    record = {
        "config": {
            "source": "R3-F",
            "corruption_level": level,
            "s": 0.0,
            "asymmetric_mode": None,
            "onset_pct": onset,
            "seed": seed
        },
        "endpoints": {
            "acc_A": float(acc_A),
            "loss_A": float(loss_A),
            "acc_B": float(acc_B),
            "loss_B": float(loss_B),
            "acc_naive_merged": float(acc_merged),
            "loss_naive_merged": float(loss_merged),
            "acc_dip": float(acc_dip)
        },
        "distances": dist_metrics,
        "curvature": curv_metrics,
        "layerwise_cka": layerwise_cka,
        "cure1_zero_shot_logit_adj": cure1_res,
        "cure2_few_shot_recalibration": {
            "acc": float(fs_acc),
            "gain": float(fs_gain),
            "recovery_ratio": float(fs_rec_ratio)
        },
        "cure3_decoupled_head_ensemble": cure3_res,
        "cure4_linear_probe_upper_bound": {
            "probe_acc": float(probe_acc),
            "probe_gap": float(probe_gap)
        }
    }

    out_file = os.path.join(output_dir, f"result_cures_R3-F_{level}_onset{onset}_seed{seed}.json")
    with open(out_file, "w") as fp:
        json.dump(record, fp, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Feature Skew Control & Falsification Runner")
    parser.add_argument("--corruption_levels", type=str, nargs="+", default=["mild", "severe"],
                        choices=["mild", "severe"], help="Feature corruption levels to evaluate")
    parser.add_argument("--onsets", type=float, nargs="+", default=[2.0, 10.0, 25.0],
                        help="Onset percentages in {2.0, 10.0, 25.0}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to evaluate")
    parser.add_argument("--total_steps", type=int, default=25000, help="Total step budget (reference = 25,000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/cache", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./runs/stage3_feature_skew", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent parallel processes on GPU")
    parser.add_argument("--quick_test", action="store_true", help="Quick smoke test with 200 steps")
    args = parser.parse_args()

    if args.quick_test:
        args.total_steps = 200
        args.onsets = [2.0]
        args.seeds = [42]
        args.corruption_levels = ["mild"]

    os.makedirs(args.output_dir, exist_ok=True)
    plan = get_feature_skew_plan(args.corruption_levels, args.onsets, args.seeds)

    print("\n" + "=" * 90)
    print(f"STAGE 3: FEATURE SKEW FALSIFICATION SUITE (Total Runs: {len(plan)})")
    print(f"Output Directory: {args.output_dir} | Max Workers: {args.max_workers} | Quick Test: {args.quick_test}")
    print("=" * 90)

    # 1. Check for already completed cure files
    pending_training = []
    for cfg in plan:
        level, onset, seed = cfg["corruption_level"], cfg["onset_pct"], cfg["seed"]
        cure_out = os.path.join(args.output_dir, f"result_cures_R3-F_{level}_onset{onset}_seed{seed}.json")
        harness_out = os.path.join(args.output_dir, f"result_R3-F_{level}_onset{onset}_seed{seed}.json")
        if os.path.exists(cure_out):
            print(f"  [Skip] Already completed: {os.path.basename(cure_out)}")
        elif os.path.exists(harness_out):
            print(f"  [Ready for Cures] Found training results: {os.path.basename(harness_out)}")
        else:
            pending_training.append(cfg)

    # 2. Train pending models
    if pending_training:
        print(f"\nLaunching {len(pending_training)} training runs across workers={args.max_workers}...\n")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_cfg = {
                executor.submit(run_single_feature_skew_training, cfg, args): cfg
                for cfg in pending_training
            }
            for future in as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                out_name, elapsed, ret_code, log_path = future.result()
                status = "SUCCESS" if ret_code == 0 else f"FAILED (code {ret_code})"
                print(f"  [{status}] {cfg['corruption_level']} onset={cfg['onset_pct']}% seed={cfg['seed']} ({elapsed:.1f}s)")
                if ret_code != 0:
                    print(f"    Check log: {log_path}")

    # 3. Evaluate Mechanistic Decomposition & Cures on all completed runs
    print("\n" + "-" * 90)
    print("Evaluating Mechanistic Decomposition & Head Cures on all Feature Skew models...")
    print("-" * 90)
    all_cure_results = []
    for cfg in plan:
        level, onset, seed = cfg["corruption_level"], cfg["onset_pct"], cfg["seed"]
        cure_out = os.path.join(args.output_dir, f"result_cures_R3-F_{level}_onset{onset}_seed{seed}.json")
        if os.path.exists(cure_out):
            with open(cure_out, "r") as fp:
                all_cure_results.append(json.load(fp))
            continue

        try:
            t0 = time.time()
            res = evaluate_feature_skew_cures(
                cfg=cfg,
                model_dir=args.output_dir,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                device_name=args.device
            )
            elapsed = time.time() - t0
            print(f"  Finished {os.path.basename(cure_out)} in {elapsed:.1f}s | "
                  f"Naive: {res['endpoints']['acc_naive_merged']*100:.2f}% -> Cure2: {res['cure2_few_shot_recalibration']['acc']*100:.2f}%")
            all_cure_results.append(res)
        except Exception as e:
            print(f"  Error evaluating cures for R3-F {level} onset {onset} seed {seed}: {e}")

    # Save summary master JSON
    summary_path = os.path.join(args.output_dir, "stage3_feature_skew_master_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(all_cure_results, fp, indent=2)

    print(f"\n--> Finished Stage 3 Feature Skew Suite! Saved summary to {summary_path}\n" + "=" * 90)


if __name__ == "__main__":
    main()
