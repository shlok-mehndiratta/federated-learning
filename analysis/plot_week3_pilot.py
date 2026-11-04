"""Analysis and Plotting Script for Week 3 (Gate G2: Baseline Anchors & Pilot Sweep).

Loads:
- ./runs/week3_pilot/week3_results.json

Generates:
- ./runs/week3_pilot/anchors_v3_v4.png
- ./runs/week3_pilot/pilot_onset_recovery.png
- Prints Gate G2 Pre-Registration Decision Summary (Branch A vs B vs C).
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


def analyze_anchors(results: List[Dict], output_dir: str):
    """Analyzes and plots Anchor V3 and Anchor V4."""
    anchors = [r for r in results if r.get("experiment", "").startswith("anchor_")]
    if not anchors:
        print("No anchor results found in dataset.")
        return

    print("\n" + "=" * 65)
    print("WEEK 3 BASELINE ANCHORS SUMMARY")
    print("=" * 65)
    print(f"{'Anchor':<15} | {'Condition':<20} | {'Barrier':<10} | {'Merged Acc':<12} | {'RTB':<6}")
    print("-" * 65)

    v3b = next((r for r in anchors if r.get("experiment") == "anchor_v3b"), None)
    v3a = next((r for r in anchors if r.get("experiment") == "anchor_v3a"), None)
    v4_i2 = next((r for r in anchors if r.get("experiment") == "anchor_v4" and r.get("isolation_level") == "I2"), None)
    v4_i0 = next((r for r in anchors if r.get("experiment") == "anchor_v4" and r.get("isolation_level") == "I0"), None)

    for name, r in [("Anchor V3b", v3b), ("Anchor V3a", v3a), ("Anchor V4 (I2)", v4_i2), ("Anchor V4 (I0)", v4_i0)]:
        if r is not None:
            cond = f"{r.get('init_type', '')} ({r.get('isolation_level')})"
            print(f"{name:<15} | {cond:<20} | {r['loss_barrier']:<10.4f} | {r['merged_acc']*100:<11.2f}% | {r['rtb']:<6}")
    print("-" * 65)

    # Plot Anchors
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Anchor V3 (Shared vs Different Init)
    if v3b is not None and v3a is not None:
        labels = ["Zhu et al. (V3b)\nDifferent Inits", "Our Setting (V3a)\nShared Init"]
        barriers = [v3b["loss_barrier"], v3a["loss_barrier"]]
        accs = [v3b["merged_acc"] * 100, v3a["merged_acc"] * 100]

        x = np.arange(len(labels))
        width = 0.35

        ax1 = axes[0]
        rects1 = ax1.bar(x - width/2, barriers, width, label='11-pt Loss Barrier', color='#d62728', alpha=0.85)
        ax1.set_ylabel('Loss Barrier (Cross-Entropy)', color='#d62728', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='#d62728')
        ax1.set_ylim(0, max(barriers) * 1.25 if max(barriers) > 0 else 1.0)

        ax1_twin = ax1.twinx()
        rects2 = ax1_twin.bar(x + width/2, accs, width, label='Merged Accuracy (%)', color='#2ca02c', alpha=0.85)
        ax1_twin.set_ylabel('Merged Top-1 Accuracy (%)', color='#2ca02c', fontsize=11)
        ax1_twin.tick_params(axis='y', labelcolor='#2ca02c')
        ax1_twin.set_ylim(0, 100)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax1.set_title("Anchor V3: Impact of Shared Starting Seed", fontsize=11, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.25)

    # Subplot 2: Anchor V4 (Randl Gossip vs Full Isolation)
    if v4_i2 is not None and v4_i0 is not None:
        labels = ["Randl Gossip (I2)\nDense Coupling", "Full Isolation (I0)\nZero Comm"]
        rtbs = [v4_i2["rtb"], v4_i0["rtb"]]
        barriers_v4 = [v4_i2["loss_barrier"], v4_i0["loss_barrier"]]

        x = np.arange(len(labels))
        width = 0.35

        ax2 = axes[1]
        rects1 = ax2.bar(x - width/2, barriers_v4, width, label='Loss Barrier', color='#ff7f0e', alpha=0.85)
        ax2.set_ylabel('Loss Barrier', color='#ff7f0e', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')

        ax2_twin = ax2.twinx()
        rects2 = ax2_twin.bar(x + width/2, rtbs, width, label='Rounds-to-Baseline (RTB)', color='#1f77b4', alpha=0.85)
        ax2_twin.set_ylabel('Recovery Rounds (RTB)', color='#1f77b4', fontsize=11)
        ax2_twin.tick_params(axis='y', labelcolor='#1f77b4')

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax2.set_title("Anchor V4: Randl Gossip vs. True Isolation", fontsize=11, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    out_plot = os.path.join(output_dir, "anchors_v3_v4.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved Anchors Plot to: {out_plot}")


def analyze_pilot(results: List[Dict], output_dir: str):
    """Analyzes and plots the 1-seed pilot sweep."""
    pilot_runs = [r for r in results if r.get("experiment") == "pilot"]
    if not pilot_runs:
        print("No pilot runs found in dataset.")
        return

    print("\n" + "=" * 70)
    print("WEEK 3 PILOT SWEEP SUMMARY (GATE G2)")
    print("=" * 70)
    print(f"{'s':<5} | {'Onset (Rnd)':<12} | {'Iso':<5} | {'Barrier':<10} | {'D_pair':<10} | {'Merged Acc':<12} | {'RTB':<6}")
    print("-" * 70)

    for r in sorted(pilot_runs, key=lambda x: (x["heterogeneity_s"], x["onset_round"], x["isolation_level"])):
        print(f"{r['heterogeneity_s']:<5.1f} | {r['onset_round']:<12} | {r['isolation_level']:<5} | {r['loss_barrier']:<10.4f} | {r['d_pair']:<10.4f} | {r['merged_acc']*100:<11.2f}% | {r['rtb']:<6}")
    print("-" * 70)

    # Plot Pilot Curves under I0
    i0_runs = [r for r in pilot_runs if r.get("isolation_level") == "I0"]
    if not i0_runs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    s_levels = sorted(list(set(r["heterogeneity_s"] for r in i0_runs)))
    colors = {0.0: "#2ca02c", 1.0: "#d62728"}

    for s in s_levels:
        subset = [r for r in i0_runs if abs(r["heterogeneity_s"] - s) < 1e-4]
        subset = sorted(subset, key=lambda x: x["onset_round"])
        onsets = [r["onset_round"] for r in subset]
        barriers = [r["loss_barrier"] for r in subset]
        rtbs = [r["rtb"] for r in subset]
        label = f"s = {s:.1f} ({'IID' if s == 0.0 else 'Disjoint'})"

        axes[0].plot(onsets, barriers, marker='o', lw=2, color=colors.get(s, '#333'), label=label)
        axes[1].plot(onsets, rtbs, marker='s', lw=2, color=colors.get(s, '#333'), label=label)

    axes[0].set_title("Loss Barrier vs. Outage Onset Round (I0)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[0].set_ylabel("11-Point Loss Barrier (CE)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Recovery Rounds (RTB) vs. Outage Onset (I0)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[1].set_ylabel("Rounds to Baseline (RTB)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_plot = os.path.join(output_dir, "pilot_onset_recovery.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved Pilot Onset Plot to: {out_plot}")

    # Gate G2 Pre-Registration Decision Rule Evaluation
    print("\n" + "=" * 70)
    print("GATE G2 DECISION RULE CHECK (PREREGISTRATION.MD):")
    print("=" * 70)

    # Check high-s isolation penalty
    s1_early = [r for r in i0_runs if abs(r["heterogeneity_s"] - 1.0) < 1e-4 and r["onset_round"] == 0]
    s1_late = [r for r in i0_runs if abs(r["heterogeneity_s"] - 1.0) < 1e-4 and r["onset_round"] >= 2]

    if s1_early and s1_late:
        early_barrier = s1_early[0]["loss_barrier"]
        late_barrier = min(r["loss_barrier"] for r in s1_late)
        early_rtb = s1_early[0]["rtb"]
        late_rtb = min(r["rtb"] for r in s1_late)

        print(f"  • s=1.0 Early Outage (k=0) : Barrier = {early_barrier:.4f} | RTB = {early_rtb}")
        print(f"  • s=1.0 Later Outage (k>=2): Min Barrier = {late_barrier:.4f} | Min RTB = {late_rtb}")

        if early_barrier > late_barrier and early_rtb >= late_rtb:
            print("\n--> [BRANCH A CONFIRMED (EXPECTED)]: Isolation hurts at high skew (s=1), but later onset restores mergeability.")
            print("--> PROCEED TO WEEKS 4-5 FULL FACTORIAL GRID!")
        else:
            print("\n--> [BRANCH B OR C DETECTED]: Review parameters before full sweep.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Week 3 Anchors & Pilot Results")
    parser.add_argument("--json_path", type=str, default="./runs/week3_pilot/week3_results.json")
    parser.add_argument("--output_dir", type=str, default="./runs/week3_pilot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_json(args.json_path)

    analyze_anchors(results, args.output_dir)
    analyze_pilot(results, args.output_dir)


if __name__ == "__main__":
    main()
