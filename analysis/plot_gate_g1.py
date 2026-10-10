"""Analysis Script for Gate G1: Validate V1 Butterfly Reproduction & V2 FL Trunk Stability.

Loads:
- ./runs/v1_butterfly/v1_results.json
- ./runs/v2_trunk/v2_results.json

Generates:
- ./runs/v1_butterfly_reproduction.png (Compares loss barrier decay across onsets)
- ./runs/v2_fl_vs_centralized.png (Compares centralized vs federated instability timescale)
- Prints Gate G1 verification table and pass/fail summary.
"""

import argparse
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def analyze_v1(v1_data: List[Dict], output_dir: str):
    """Analyzes and plots Probe V1 results."""
    onsets = sorted(list(set(d["onset_pct"] for d in v1_data)))
    means = []
    stds = []
    d_pairs = []

    print("\n" + "=" * 60)
    print("PROBE V1: BUTTERFLY EFFECT REPRODUCTION SUMMARY")
    print("=" * 60)
    print(f"{'Onset (%)':<12} | {'Mean Barrier':<15} | {'Std Dev':<10} | {'Mean D_pair':<12} | {'Merged Acc':<12}")
    print("-" * 60)

    for o in onsets:
        subset = [d for d in v1_data if abs(d["onset_pct"] - o) < 1e-4]
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

    print("-" * 60)

    # Plot V1 Barrier vs Onset
    plt.figure(figsize=(7, 5))
    plt.errorbar(onsets, means, yerr=stds, fmt='-o', color='#1f77b4', lw=2, capsize=5, label='Our Harness (ResNet-20 LN)')
    plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Published Transition (~2%)')
    plt.title("Probe V1: Centralized Loss Barrier vs. Perturbation Onset", fontsize=12, fontweight='bold')
    plt.xlabel("Perturbation Onset (% of training)", fontsize=11)
    plt.ylabel("11-Point Loss Barrier (CE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, "v1_butterfly_reproduction.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Plot saved to: {out_plot}")

    # Check Gate G1 condition
    initial_barrier = means[0] if len(means) > 0 else 0.0
    late_barrier = means[-1] if len(means) > 0 else 0.0

    print("\nGate G1 Check for Probe V1:")
    if initial_barrier > 0.3 and late_barrier < 0.1:
        print("--> [PASS] V1 successfully reproduced the sharp chaotic-to-stable transition!")
    else:
        print(f"--> [REVIEW NEEDED] Initial barrier={initial_barrier:.4f}, Late barrier={late_barrier:.4f}")


def analyze_v2(v2_data: List[Dict], output_dir: str):
    """Analyzes and plots Probe V2 results."""
    onsets = sorted(list(set(d["onset_pct"] for d in v2_data)))
    means = []
    stds = []

    print("\n" + "=" * 60)
    print("PROBE V2: FEDERATED TRUNK STABILITY (s=0) SUMMARY")
    print("=" * 60)
    print(f"{'Onset (%)':<12} | {'FL Barrier':<15} | {'Std Dev':<10} | {'D_pair':<12} | {'Merged Acc':<12}")
    print("-" * 60)

    for o in onsets:
        subset = [d for d in v2_data if abs(d["onset_pct"] - o) < 1e-4]
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

    print("-" * 60)

    plt.figure(figsize=(7, 5))
    plt.errorbar(onsets, means, yerr=stds, fmt='-s', color='#2ca02c', lw=2, capsize=5, label='FedAvg Trunk (s=0)')
    plt.title("Probe V2: Federated Trunk Loss Barrier vs. Onset", fontsize=12, fontweight='bold')
    plt.xlabel("Fork Onset (% of training)", fontsize=11)
    plt.ylabel("11-Point Loss Barrier (CE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, "v2_fl_stability_boundary.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Plot saved to: {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Gate G1 Results")
    parser.add_argument("--v1_json", type=str, default="./runs/v1_butterfly/v1_results.json")
    parser.add_argument("--v2_json", type=str, default="./runs/v2_trunk/v2_results.json")
    parser.add_argument("--output_dir", type=str, default="./runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.v1_json):
        v1_data = load_json(args.v1_json)
        analyze_v1(v1_data, args.output_dir)
    else:
        print(f"V1 results not yet found at {args.v1_json}. Skipping V1 analysis.")

    if os.path.exists(args.v2_json):
        v2_data = load_json(args.v2_json)
        analyze_v2(v2_data, args.output_dir)
    else:
        print(f"V2 results not yet found at {args.v2_json}. Skipping V2 analysis.")


if __name__ == "__main__":
    main()
