"""
Stage 4: Autonomous Triage Reconnection Diagnostic Runner.

Trains and validates the empirical Decision-Tree meta-predictor on telemetry collected
across experimental stages (Stage 1, Stage 2, and Stage 3):
- Maps observable telemetry (onset k, D_pair, D_head/D_body, Penultimate CKA) to optimal merge action.
- Evaluates 5-fold cross-validation accuracy.
- Exports human-readable if-then decision rules for edge parameter servers.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.triage_decision_tree import (
    PureNumPyDecisionTree,
    compile_meta_predictor_dataset,
    evaluate_cross_validation,
    FEATURE_NAMES,
    ACTION_NAMES
)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Autonomous Triage Meta-Predictor")
    parser.add_argument("--stage1_dir", type=str, default="./runs/stage1_science")
    parser.add_argument("--stage2_dir", type=str, default="./runs/stage2_head_cures")
    parser.add_argument("--stage3_dir", type=str, default="./runs/stage3_feature_skew")
    parser.add_argument("--output_dir", type=str, default="./runs/stage4_meta_predictor")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum tree depth for interpretability")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("STAGE 4: AUTONOMOUS TRIAGE RECONNECTION DIAGNOSTIC META-PREDICTOR")
    print("=" * 80)

    # 1. Compile empirical dataset
    X, y, records = compile_meta_predictor_dataset(args.stage1_dir, args.stage2_dir, args.stage3_dir)
    print(f"Loaded {len(records)} empirical records across conditions.")
    if len(records) == 0:
        print("Error: No records found. Ensure Stage 2 / Stage 3 results are present.")
        return

    # 2. 5-Fold Cross Validation
    cv_acc = evaluate_cross_validation(X, y, k_folds=5, seed=42)
    print(f"\n--> 5-Fold Stratified Cross-Validation Accuracy: {cv_acc * 100:.2f}%\n")

    # 3. Train final interpretable decision tree on full empirical dataset
    tree = PureNumPyDecisionTree(max_depth=args.max_depth)
    tree.fit(X, y)

    # 4. Export ASCII Decision Rules
    rules_text = tree.export_ascii_rules(feature_names=FEATURE_NAMES, action_names=ACTION_NAMES)
    print(rules_text)

    # 5. Save rules and artifacts
    output_meta = {
        "dataset_size": len(records),
        "cv_accuracy": cv_acc,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "rules_text": rules_text
    }
    meta_path = os.path.join(args.output_dir, "meta_predictor_summary.json")
    with open(meta_path, "w") as fp:
        json.dump(output_meta, fp, indent=2)

    print(f"\nSaved Meta-Predictor Model Summary to {meta_path}\n" + "=" * 80)


if __name__ == "__main__":
    main()
