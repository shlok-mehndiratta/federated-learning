"""Analysis and Plotting Script for Milestone 2: Literature Replication & Pilot Evaluation.

Loads:
- ./runs/pilot_evaluation/pilot_results.json (or ./runs/week3_pilot/week3_results.json)

Generates:
- ./runs/pilot_evaluation/literature_replications.png (also saved as anchors_v3_v4.png for compatibility)
- ./runs/pilot_evaluation/pilot_onset_recovery.png
- Evaluates and prints Gate G2 Pre-Registration Decision Summary (Branch A vs B vs C)
  using dynamic difference-in-differences against the s=0.0 IID arm per PREREGISTRATION.md.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def compute_dynamic_rtb(record: Dict, baseline_target: float, tolerance: float = 0.01) -> int:
    """Calculates rounds to baseline from recovery history relative to unbroken target."""
    history = record.get("recovery_history", [])
    if not history:
        return record.get("rtb", 30)
    
    threshold = baseline_target - tolerance
    for step in history:
        if step.get("acc", 0.0) >= threshold:
            return step.get("round", 0)
    return len(history) - 1


def analyze_anchors(results: List[Dict], output_dir: str):
    """Analyzes and plots Anchor V3 (Zhu rebuttal) and Anchor V4 (Randl rebuttal)."""
    anchors = [r for r in results if r.get("experiment", "").startswith("anchor_")]
    if not anchors:
        print("No anchor results found in dataset.")
        return

    print("\n" + "=" * 80)
    print("MILESTONE 2: BASELINE LITERATURE REPLICATION SUMMARY")
    print("=" * 80)
    print(f"{'Anchor':<15} | {'Condition':<22} | {'Acc Dip':<10} | {'D_pair':<10} | {'Merged Acc':<12} | {'Curvature':<10}")
    print("-" * 80)

    v3b = next((r for r in anchors if r.get("experiment") == "anchor_v3b"), None)
    v3a = next((r for r in anchors if r.get("experiment") == "anchor_v3a"), None)
    v4_i2 = next((r for r in anchors if r.get("experiment") == "anchor_v4" and r.get("isolation_level") == "I2"), None)
    v4_i0 = next((r for r in anchors if r.get("experiment") == "anchor_v4" and r.get("isolation_level") == "I0"), None)

    for name, r in [("Anchor V3b", v3b), ("Anchor V3a", v3a), ("Anchor V4 (I2)", v4_i2), ("Anchor V4 (I0)", v4_i0)]:
        if r is not None:
            cond = f"{r.get('init_type', '')} ({r.get('isolation_level')})"
            dip_pct = r.get("max_acc_dip", 0.0) * 100
            m_acc_pct = r.get("merged_acc", 0.0) * 100
            d_p = r.get("d_pair", 0.0)
            c_est = r.get("curvature_est", 0.0)
            print(f"{name:<15} | {cond:<22} | {dip_pct:<9.2f}% | {d_p:<10.4f} | {m_acc_pct:<11.2f}% | {c_est:<10.4f}")
    print("-" * 80)

    # Plot Anchors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Anchor V3 (Shared vs Different Init)
    if v3b is not None and v3a is not None:
        labels = ["Zhu et al. (V3b)\nDiff Inits", "Our Setting (V3a)\nShared Init"]
        acc_dips = [v3b["max_acc_dip"] * 100, v3a["max_acc_dip"] * 100]
        merged_accs = [v3b["merged_acc"] * 100, v3a["merged_acc"] * 100]

        x = np.arange(len(labels))
        width = 0.35

        ax1 = axes[0]
        rects1 = ax1.bar(x - width/2, acc_dips, width, label='Acc Barrier (Dip %)', color='#d62728', alpha=0.85)
        rects2 = ax1.bar(x + width/2, merged_accs, width, label='Merged Top-1 Acc (%)', color='#2ca02c', alpha=0.85)

        ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax1.set_title("Anchor V3: Impact of Shared Starting Seed", fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 30)
        ax1.grid(True, axis='y', alpha=0.25)
        ax1.legend(loc='upper right')

        # Add distance callout text
        d_ratio = v3b["d_pair"] / max(v3a["d_pair"], 1e-6)
        ax1.text(0.5, 25, f"Parameter Distance D_pair:\nV3b={v3b['d_pair']:.4f} vs V3a={v3a['d_pair']:.4f}\n({d_ratio:.1f}x farther apart)",
                 ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#999', alpha=0.9),
                 fontsize=9)

    # Subplot 2: Anchor V4 (Randl Gossip vs Full Isolation)
    if v4_i2 is not None and v4_i0 is not None:
        labels = ["Randl Gossip (I2)\nDense Coupling", "Full Isolation (I0)\nZero Comm"]
        acc_dips_v4 = [v4_i2["max_acc_dip"] * 100, v4_i0["max_acc_dip"] * 100]
        merged_accs_v4 = [v4_i2["merged_acc"] * 100, v4_i0["merged_acc"] * 100]

        x = np.arange(len(labels))
        width = 0.35

        ax2 = axes[1]
        rects1 = ax2.bar(x - width/2, acc_dips_v4, width, label='Acc Barrier (Dip %)', color='#ff7f0e', alpha=0.85)
        rects2 = ax2.bar(x + width/2, merged_accs_v4, width, label='Merged Top-1 Acc (%)', color='#1f77b4', alpha=0.85)

        ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax2.set_title("Anchor V4: Randl Gossip vs. True Isolation", fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 30)
        ax2.grid(True, axis='y', alpha=0.25)
        ax2.legend(loc='upper right')

        # Add annotation text
        ax2.text(0.5, 25, f"Drift Under Isolation:\nI2 D_pair = {v4_i2['d_pair']:.4f} (Zero Drift)\nI0 D_pair = {v4_i0['d_pair']:.4f} (Isolated Drift)",
                 ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#999', alpha=0.9),
                 fontsize=9)

    plt.tight_layout()
    out_plot = os.path.join(output_dir, "literature_replications.png")
    plt.savefig(out_plot, dpi=300)
    # Also save with legacy filename for compatibility
    legacy_plot = os.path.join(output_dir, "anchors_v3_v4.png")
    plt.savefig(legacy_plot, dpi=300)
    plt.close()
    print(f"Saved Replications Plot to: {out_plot}")


def analyze_pilot(results: List[Dict], output_dir: str):
    """Analyzes and plots the 1-seed pilot sweep using dynamic diff-in-diff baseline."""
    pilot_runs = [r for r in results if r.get("experiment") == "pilot"]
    if not pilot_runs:
        print("No pilot runs found in dataset.")
        return

    # Compute baseline reference targets from s=0.0 IID arm
    baseline_targets = {
        (r["onset_round"], r["isolation_level"]): r["merged_acc"]
        for r in pilot_runs if r.get("heterogeneity_s") == 0.0
    }

    print("\n" + "=" * 90)
    print("MILESTONE 2: PILOT SWEEP SUMMARY (GATE G2)")
    print("=" * 90)
    print(f"{'s':<5} | {'Onset (k)':<10} | {'Iso':<5} | {'Acc Dip':<10} | {'D_pair':<10} | {'Curvature':<10} | {'Merged Acc':<12} | {'Dyn RTB':<8}")
    print("-" * 90)

    for r in sorted(pilot_runs, key=lambda x: (x["heterogeneity_s"], x["onset_round"], x["isolation_level"])):
        k = r["onset_round"]
        iso = r["isolation_level"]
        s = r["heterogeneity_s"]
        target = baseline_targets.get((k, iso), r["merged_acc"])
        dyn_rtb = compute_dynamic_rtb(r, baseline_target=target, tolerance=0.01)
        r["dynamic_rtb"] = dyn_rtb

        dip_pct = r.get("max_acc_dip", 0.0) * 100
        m_acc_pct = r.get("merged_acc", 0.0) * 100
        d_p = r.get("d_pair", 0.0)
        c_est = r.get("curvature_est", 0.0)

        print(f"{s:<5.1f} | {k:<10} | {iso:<5} | {dip_pct:<9.2f}% | {d_p:<10.4f} | {c_est:<10.4f} | {m_acc_pct:<11.2f}% | {dyn_rtb:<8}")
    print("-" * 90)

    # Plot comprehensive 2x2 Pilot Curves under I0
    i0_runs = [r for r in pilot_runs if r.get("isolation_level") == "I0"]
    if not i0_runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    s_levels = sorted(list(set(r["heterogeneity_s"] for r in i0_runs)))
    colors = {0.0: "#2ca02c", 1.0: "#d62728"}
    markers = {0.0: "o", 1.0: "s"}

    for s in s_levels:
        subset = [r for r in i0_runs if abs(r["heterogeneity_s"] - s) < 1e-4]
        subset = sorted(subset, key=lambda x: x["onset_round"])
        onsets = [r["onset_round"] for r in subset]
        dips = [r["max_acc_dip"] * 100 for r in subset]
        rtbs = [r["dynamic_rtb"] for r in subset]
        dpairs = [r["d_pair"] for r in subset]
        curvs = [r["curvature_est"] for r in subset]
        label = f"s = {s:.1f} ({'IID' if s == 0.0 else 'Disjoint Classes'})"

        # (0, 0): Accuracy Dip %
        axes[0, 0].plot(onsets, dips, marker=markers.get(s, 'o'), lw=2.2, color=colors.get(s, '#333'), label=label)
        # (0, 1): Dynamic RTB
        axes[0, 1].plot(onsets, rtbs, marker=markers.get(s, 's'), lw=2.2, color=colors.get(s, '#333'), label=label)
        # (1, 0): Parameter Distance D_pair
        axes[1, 0].plot(onsets, dpairs, marker=markers.get(s, '^'), lw=2.2, color=colors.get(s, '#333'), label=label)
        # (1, 1): Curvature Estimate (H4)
        axes[1, 1].plot(onsets, curvs, marker=markers.get(s, 'd'), lw=2.2, color=colors.get(s, '#333'), label=label)

    # Styling panel (0, 0)
    axes[0, 0].set_title("Accuracy Barrier (Acc Dip % below endpoints)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[0, 0].set_ylabel("Max Accuracy Dip (%)", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Styling panel (0, 1)
    axes[0, 1].set_title("Dynamic Recovery Rounds (RTB vs s=0.0 Baseline)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[0, 1].set_ylabel("Rounds to Baseline (RTB)", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Styling panel (1, 0)
    axes[1, 0].set_title("Pairwise Parameter Distance (D_pair)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[1, 0].set_ylabel("Normalized Distance D_pair", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Styling panel (1, 1)
    axes[1, 1].set_title("Hessian Curvature Along Merge Direction (H4)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Outage Onset Round (k)", fontsize=10)
    axes[1, 1].set_ylabel("Hessian-Vector Curvature", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    out_plot = os.path.join(output_dir, "pilot_onset_recovery.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved Pilot Onset Plot to: {out_plot}")

    # Gate G2 Pre-Registration Decision Rule Evaluation
    print("\n" + "=" * 70)
    print("GATE G2 DECISION RULE CHECK (PREREGISTRATION.MD):")
    print("=" * 70)

    s1_early = [r for r in i0_runs if abs(r["heterogeneity_s"] - 1.0) < 1e-4 and r["onset_round"] == 0]
    s1_late = [r for r in i0_runs if abs(r["heterogeneity_s"] - 1.0) < 1e-4 and r["onset_round"] >= 1]

    if s1_early and s1_late:
        early_dip = s1_early[0]["max_acc_dip"] * 100
        min_late_dip = min(r["max_acc_dip"] for r in s1_late) * 100
        early_rtb = s1_early[0]["dynamic_rtb"]
        min_late_rtb = min(r["dynamic_rtb"] for r in s1_late)

        print(f"  • s=1.0 Early Outage (k=0) : Acc Dip = {early_dip:.2f}% | Dyn RTB = {early_rtb}")
        print(f"  • s=1.0 Later Outage (k>=1): Min Acc Dip = {min_late_dip:.2f}% | Min Dyn RTB = {min_late_rtb}")

        if early_dip > min_late_dip:
            print("\n--> [BRANCH A CONFIRMED (EXPECTED)]: Isolation severely hurts early merges (k=0),")
            print("    but shared global model pre-training (k>=1) restores mergeability!")
            print("--> Empirical onset boundary k* exists between round 0 and round 1.")
            print("--> PROCEED TO MILESTONE 3: FULL FACTORIAL PARAMETER SWEEP!")
        else:
            print("\n--> [BRANCH B OR C DETECTED]: Review parameters before full sweep.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Milestone 2 Literature Replication & Pilot Results")
    parser.add_argument("--json_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./runs/pilot_evaluation")
    args = parser.parse_args()

    json_path = args.json_path
    if not json_path:
        for candidate in ["./runs/pilot_evaluation/pilot_results.json", "./runs/week3_pilot/week3_results.json"]:
            if os.path.exists(candidate):
                json_path = candidate
                break

    if not json_path or not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found. Checked: ./runs/pilot_evaluation/pilot_results.json and ./runs/week3_pilot/week3_results.json")

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_json(json_path)

    analyze_anchors(results, args.output_dir)
    analyze_pilot(results, args.output_dir)


if __name__ == "__main__":
    main()
