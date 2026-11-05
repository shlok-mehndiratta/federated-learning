"""Analysis Script for Milestone 1: Centralized vs. Federated Stability Boundary Analysis.

Loads:
- ./runs/centralized_stability/centralized_results.json (or ./runs/v1_butterfly/v1_results.json)
- ./runs/federated_stability/federated_results.json (or ./runs/v2_trunk/v2_results.json)

Generates:
- ./runs/centralized_stability_boundary.png (Centralized SGD loss barrier decay)
- ./runs/federated_stability_boundary.png (20-client FedAvg stability timescale)
- Prints Milestone 1 verification table and pass/fail summary.
"""

import argparse
import json
import os
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def analyze_centralized(data: List[Dict], output_dir: str):
    """Analyzes and plots centralized SGD stability boundary."""
    onsets = sorted(list(set(d["onset_pct"] for d in data)))
    means = []
    stds = []
    d_pairs = []

    print("\n" + "=" * 65)
    print("MILESTONE 1: CENTRALIZED SGD STABILITY EVALUATION")
    print("=" * 65)
    print(f"{'Onset (%)':<12} | {'Mean Barrier':<15} | {'Std Dev':<10} | {'Mean D_pair':<12} | {'Merged Acc':<12}")
    print("-" * 65)

    for o in onsets:
        subset = [d for d in data if abs(d["onset_pct"] - o) < 1e-4]
        barriers = [d["loss_barrier"] for d in subset]
        dp = [d["d_pair"] for d in subset]
        accs = [d["merged_acc"] * 100 for d in subset]

        mean_b = float(np.mean(barriers))
        std_b = float(np.std(barriers))
        mean_dp = float(np.mean(dp))
        mean_acc = float(np.mean(accs))

        means.append(mean_b)
        stds.append(std_b)
        d_pairs.append(mean_dp)

        print(f"{o:<12.2f} | {mean_b:<15.4f} | {std_b:<10.4f} | {mean_dp:<12.4f} | {mean_acc:<11.2f}%")

    print("-" * 65)

    # Plot Barrier vs Onset
    plt.figure(figsize=(7, 5))
    plt.errorbar(onsets, means, yerr=stds, fmt='-o', color='#1f77b4', lw=2, capsize=5, label='Centralized SGD (ResNet-20 LN)')
    plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Literature Transition (~2%)')
    plt.title("Centralized SGD Loss Barrier vs. Perturbation Onset", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Onset (% of training)", fontsize=11)
    plt.ylabel("11-Point Loss Barrier (CE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, "centralized_stability_boundary.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Plot saved to: {out_plot}")

    # Check Milestone 1 condition
    initial_barrier = means[0] if len(means) > 0 else 0.0
    late_barrier = means[-1] if len(means) > 0 else 0.0

    print("\nMilestone 1 Check (Centralized):")
    if initial_barrier > 0.3 and late_barrier < 0.1:
        print("--> [PASS] Successfully reproduced the sharp chaotic-to-stable transition!")
    else:
        print(f"--> [REVIEW NEEDED] Initial barrier={initial_barrier:.4f}, Late barrier={late_barrier:.4f}")


def analyze_federated(data: List[Dict], output_dir: str):
    """Analyzes and plots 20-client FedAvg stability boundary."""
    onsets = sorted(list(set(d["onset_pct"] for d in data)))
    means = []
    stds = []

    print("\n" + "=" * 65)
    print("MILESTONE 1: FEDERATED STABILITY EVALUATION (s=0.0)")
    print("=" * 65)
    print(f"{'Onset (%)':<12} | {'FL Barrier':<15} | {'Std Dev':<10} | {'D_pair':<12} | {'Merged Acc':<12}")
    print("-" * 65)

    for o in onsets:
        subset = [d for d in data if abs(d["onset_pct"] - o) < 1e-4]
        barriers = [d["loss_barrier"] for d in subset]
        dp = [d["d_pair"] for d in subset]
        accs = [d["merged_acc"] * 100 for d in subset]

        mean_b = float(np.mean(barriers))
        std_b = float(np.std(barriers))
        mean_dp = float(np.mean(dp))
        mean_acc = float(np.mean(accs))

        means.append(mean_b)
        stds.append(std_b)

        print(f"{o:<12.2f} | {mean_b:<15.4f} | {std_b:<10.4f} | {mean_dp:<12.4f} | {mean_acc:<11.2f}%")

    print("-" * 65)

    plt.figure(figsize=(7, 5))
    plt.errorbar(onsets, means, yerr=stds, fmt='-s', color='#2ca02c', lw=2, capsize=5, label='FedAvg Global Model (s=0)')
    plt.title("Federated Learning Loss Barrier vs. Onset", fontsize=12, fontweight='bold')
    plt.xlabel("Fork Onset (% of training)", fontsize=11)
    plt.ylabel("11-Point Loss Barrier (CE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, "federated_stability_boundary.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Plot saved to: {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Milestone 1 Stability Results")
    parser.add_argument("--centralized_json", type=str, default="")
    parser.add_argument("--federated_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find centralized results path
    cent_path = args.centralized_json
    if not cent_path:
        for candidate in ["./runs/centralized_stability/centralized_results.json", "./runs/v1_butterfly/v1_results.json"]:
            if os.path.exists(candidate):
                cent_path = candidate
                break

    if cent_path and os.path.exists(cent_path):
        data = load_json(cent_path)
        analyze_centralized(data, args.output_dir)
    else:
        print("Centralized stability results not found. Skipping.")

    # Find federated results path
    fed_path = args.federated_json
    if not fed_path:
        for candidate in ["./runs/federated_stability/federated_results.json", "./runs/v2_trunk/v2_results.json"]:
            if os.path.exists(candidate):
                fed_path = candidate
                break

    if fed_path and os.path.exists(fed_path):
        data = load_json(fed_path)
        analyze_federated(data, args.output_dir)
    else:
        print("Federated stability results not found. Skipping.")


if __name__ == "__main__":
    main()
