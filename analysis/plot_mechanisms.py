"""Visualization and mechanism analysis script for Milestone 4.

Generates 4 high-resolution mechanism figures:
1. layer_wise_barrier_profile.png: Barrier contribution by stage across conditions.
2. cka_similarity_heatmap.png: Stage-by-stage Linear CKA representation alignment.
3. class_confusion_attribution.png: Error distribution, per-class recall, and cohort interference.
4. acceleration_comparison.png: Post-merge recovery acceleration (Re-Basin & Momentum Reset).
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: str) -> List[Dict]:
    """Loads mechanism results JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def plot_layer_wise_barrier_profile(records: List[Dict], output_dir: str):
    """Figure 1: Layer-Wise Barrier Decomposition across conditions."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    stages = ["init", "stage1", "stage2", "stage3", "fc"]
    stage_labels = ["Init Conv", "Stage 1", "Stage 2", "Stage 3", "FC Head"]

    # Focus on representative conditions
    target_conds = [
        ("Failing_Early_Outage", "Failing (k=0, s=1.0, I0)", "#D32F2F"),
        ("Onset_Protected", "Onset Protected (k=2, s=1.0, I0)", "#1976D2"),
        ("Coupling_Protected", "Coupled (k=0, s=1.0, I2)", "#388E3C"),
        ("IID_Control", "IID Control (s=0.0)", "#757575"),
    ]

    cond_data = {}
    for r in records:
        cname = r["condition_name"]
        if cname not in cond_data:
            cond_data[cname] = r["probe1_layer_barrier"]["stage_acc_dips"]

    x = np.arange(len(stages))
    width = 0.2

    for i, (cname, label, color) in enumerate(target_conds):
        if cname in cond_data:
            dips = [cond_data[cname].get(s, 0.0) * 100 for s in stages]
            offset = (i - 1.5) * width
            ax.bar(x + offset, dips, width, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Accuracy Barrier Dip (%)", fontsize=12, fontweight="bold")
    ax.set_title("Layer-Wise Barrier Decomposition: Where Does the Barrier Live?", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "layer_wise_barrier_profile.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cka_similarity_heatmaps(records: List[Dict], output_dir: str):
    """Figure 2: Linear CKA Representation Alignment Heatmaps."""
    target_conds = [
        ("Failing_Early_Outage", "Failing Outage (k=0, s=1.0, I0)"),
        ("Onset_Protected", "Onset Protected (k=2, s=1.0, I0)"),
        ("Coupling_Protected", "Coupled Outage (k=0, s=1.0, I2)"),
    ]

    fig, axes = plt.subplots(1, len(target_conds), figsize=(16, 5))
    if len(target_conds) == 1:
        axes = [axes]

    stage_labels = ["Init", "Stage 1", "Stage 2", "Stage 3", "Pool", "Logits"]

    for idx, (cname, title) in enumerate(target_conds):
        ax = axes[idx]
        mat = None
        for r in records:
            if r["condition_name"] == cname:
                mat = np.array(r["probe2_cka"]["matrix"])
                break

        if mat is None:
            mat = np.zeros((len(stage_labels), len(stage_labels)))

        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(stage_labels)))
        ax.set_yticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(stage_labels if idx == 0 else [], fontsize=9)
        if idx == 0:
            ax.set_ylabel("Model A Stages", fontsize=11, fontweight="bold")
        ax.set_xlabel("Model B Stages", fontsize=11, fontweight="bold")

        # Overlay numeric values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val < 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Linear CKA Similarity", fontsize=11, fontweight="bold")

    plt.suptitle("Representational Alignment Across Network Stages (Linear CKA)", fontsize=14, fontweight="bold", y=0.98)

    out_path = os.path.join(output_dir, "cka_similarity_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_class_confusion_attribution(records: List[Dict], output_dir: str):
    """Figure 3: Class Confusion and Error Attribution under Disjoint Outage."""
    failing_record = None
    for r in records:
        if r["condition_name"] == "Failing_Early_Outage":
            failing_record = r
            break

    if failing_record is None and len(records) > 0:
        failing_record = records[0]

    if failing_record is None:
        print("Warning: No records found for class confusion attribution.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Confusion matrix of merged model
    ax1 = axes[0]
    conf = np.array(failing_record["probe3_attribution"]["conf_merged"]["confusion_matrix"])
    # Row normalize to get recall percentages
    row_sums = np.maximum(conf.sum(axis=1, keepdims=True), 1)
    norm_conf = conf / row_sums * 100.0

    im = ax1.imshow(norm_conf, cmap="Blues", vmin=0, vmax=100)
    ax1.set_title("Merged Model Confusion Matrix ($W_{mid}$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))

    # Draw boundary dividing Group A (classes 0-4) and Group B (classes 5-9)
    ax1.axhline(4.5, color="red", linestyle="--", linewidth=1.5, label="Cohort Boundary")
    ax1.axvline(4.5, color="red", linestyle="--", linewidth=1.5)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="True Class Recall (%)")

    # Panel 2: Per-class recall comparison (Model A vs Model B vs Merged)
    ax2 = axes[1]
    classes = np.arange(10)
    width = 0.25

    rec_A = np.array(failing_record["probe3_attribution"]["conf_A"]["per_class_recall"]) * 100
    rec_B = np.array(failing_record["probe3_attribution"]["conf_B"]["per_class_recall"]) * 100
    rec_M = np.array(failing_record["probe3_attribution"]["conf_merged"]["per_class_recall"]) * 100

    ax2.bar(classes - width, rec_A, width, label="Model A (Classes 0-4)", color="#1976D2", alpha=0.85)
    ax2.bar(classes, rec_B, width, label="Model B (Classes 5-9)", color="#D32F2F", alpha=0.85)
    ax2.bar(classes + width, rec_M, width, label="Merged Model $W_{mid}$", color="#388E3C", alpha=0.85)

    ax2.axvline(4.5, color="black", linestyle=":", linewidth=1.5)
    ax2.text(2.0, 85, "Group A Training Domain\n(Classes 0-4)", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#BBDEFB", alpha=0.7))
    ax2.text(7.0, 85, "Group B Training Domain\n(Classes 5-9)", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", alpha=0.7))

    ax2.set_xlabel("CIFAR-10 Class Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Class Recall / Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Class Accuracy: Catastrophic Forgetting & Merging Interference", fontsize=12, fontweight="bold")
    ax2.set_xticks(classes)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "class_confusion_attribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_acceleration_comparison(records: List[Dict], output_dir: str):
    """Figure 4: Post-Merge Recovery Acceleration (Re-Basin & Momentum Reset)."""
    failing_record = None
    for r in records:
        if r["condition_name"] == "Failing_Early_Outage":
            failing_record = r
            break

    if failing_record is None and len(records) > 0:
        failing_record = records[0]

    if failing_record is None:
        print("Warning: No records found for acceleration comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    accel = failing_record["probe4_acceleration"]
    target_acc = failing_record["target_baseline_acc"] * 100.0

    # Panel 1: Re-Basin Permutation vs Standard Merge Recovery
    ax1 = axes[0]
    std_hist = accel["standard_recovery"]["history"]
    reb_hist = accel["rebasin_recovery"]["history"]

    std_rounds = [h["round"] for h in std_hist]
    std_accs = [h["acc"] * 100.0 for h in std_hist]

    reb_rounds = [h["round"] for h in reb_hist]
    reb_accs = [h["acc"] * 100.0 for h in reb_hist]

    ax1.plot(std_rounds, std_accs, "o-", color="#D32F2F", linewidth=2.0, label=f"Standard FedAvg Merge (RTB={accel['standard_recovery']['rtb']} rds)")
    ax1.plot(reb_rounds, reb_accs, "s-", color="#388E3C", linewidth=2.0, label=f"Git Re-Basin Permuted Merge (RTB={accel['rebasin_recovery']['rtb']} rds)")
    ax1.axhline(target_acc, color="black", linestyle="--", label=f"Target Baseline ({target_acc:.1f}%)")

    ax1.set_xlabel("Post-Merge Recovery Rounds", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Global Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Intervention A: Git Re-Basin Weight Matching", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9.5)

    # Panel 2: Momentum Reset vs Retained Momentum
    ax2 = axes[1]
    mom_ret_hist = accel["momentum_retained_recovery"]["history"]
    mom_res_hist = accel["momentum_reset_recovery"]["history"]

    ret_rounds = [h["round"] for h in mom_ret_hist]
    ret_accs = [h["acc"] * 100.0 for h in mom_ret_hist]

    res_rounds = [h["round"] for h in mom_res_hist]
    res_accs = [h["acc"] * 100.0 for h in mom_res_hist]

    ax2.plot(ret_rounds, ret_accs, "^-", color="#F57C00", linewidth=2.0, label=f"Momentum Retained (RTB={accel['momentum_retained_recovery']['rtb']} rds)")
    ax2.plot(res_rounds, res_accs, "D-", color="#1976D2", linewidth=2.0, label=f"Momentum Reset ($m=0$) (RTB={accel['momentum_reset_recovery']['rtb']} rds)")
    ax2.axhline(target_acc, color="black", linestyle="--", label=f"Target Baseline ({target_acc:.1f}%)")

    ax2.set_xlabel("Post-Merge Recovery Rounds", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Global Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Intervention B: Post-Merge Client Momentum Reset", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9.5)

    plt.suptitle("Post-Merge Healing Acceleration: Overcoming Early Outage Collapse", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "acceleration_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Milestone 4 Mechanism & Acceleration Plots")
    parser.add_argument("--json_path", type=str, default="./runs/mechanism_sweep/mechanism_results.json", help="Path to mechanism results JSON")
    parser.add_argument("--output_dir", type=str, default="./runs/mechanism_sweep", help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_results(args.json_path)
    print(f"Loaded {len(records)} mechanism condition records.")

    plot_layer_wise_barrier_profile(records, args.output_dir)
    plot_cka_similarity_heatmaps(records, args.output_dir)
    plot_class_confusion_attribution(records, args.output_dir)
    plot_acceleration_comparison(records, args.output_dir)
    print("All Milestone 4 figures generated successfully!")


if __name__ == "__main__":
    main()
