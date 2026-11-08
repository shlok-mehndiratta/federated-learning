"""
Stage 1 Factorial Divergence Runner (Matched Design Protocol).
Executes Rung 2b (Disjoint IID control), Rung 3-L (Label Skew), and Asymmetric Divergence.
Saves full trajectory checkpoints every 1% of branch steps to enable D_pair-matched contrasts.

Features:
- Idempotent: checks for existing JSON results in --output_dir and skips completed runs.
- Modular suites: --suite core (quick primary matrix), --suite full, --suite r2b, --suite r3l.
- Complete metric logging: 11-point barrier, D_pair, body/head decomposition, CKA, probe gap,
  few-shot head gain, head share, cumulative ZipIt prefixes, logit-offset fidelity.
"""

import argparse
import itertools
import os
import time
from typing import List, Tuple

import torch

from experiments.science_harness import run_science_experiment


def get_experiment_plan(suite: str, seeds: List[int]) -> List[dict]:
    """Builds the list of experimental conditions according to the requested suite."""
    plan = []

    if suite in ("core", "r2b", "full"):
        # Rung 2b: Disjoint IID half-split control
        onsets_r2b = [2.0, 10.0, 25.0] if suite == "core" else [2.0, 5.0, 10.0, 20.0, 40.0]
        for onset, seed in itertools.product(onsets_r2b, seeds):
            plan.append({
                "source": "R2b",
                "s": 0.0,
                "asymmetric_mode": None,
                "onset_pct": onset,
                "seed": seed
            })

    if suite in ("core", "r3l", "full"):
        # Rung 3-L: Label Skew
        if suite == "core":
            s_levels = [0.8, 1.0]
            onsets_r3l = [2.0, 10.0, 25.0]
        else:
            s_levels = [0.5, 0.8, 0.95, 1.0]
            onsets_r3l = [2.0, 5.0, 10.0, 20.0, 40.0]

        for s, onset, seed in itertools.product(s_levels, onsets_r3l, seeds):
            plan.append({
                "source": "R3-L",
                "s": s,
                "asymmetric_mode": None,
                "onset_pct": onset,
                "seed": seed
            })

    if suite in ("core", "asym", "full"):
        # Asymmetric Mode 1: Group A s=0.8, Group B balanced IID
        onsets_asym = [2.0, 10.0, 25.0] if suite == "core" else [2.0, 5.0, 10.0, 20.0, 40.0]
        for onset, seed in itertools.product(onsets_asym, seeds):
            plan.append({
                "source": "R3-L",
                "s": 0.8,
                "asymmetric_mode": "A_skew_B_iid",
                "onset_pct": onset,
                "seed": seed
            })

    return plan


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Factorial Divergence Runner")
    parser.add_argument("--suite", type=str, default="core", choices=["core", "full", "r2b", "r3l", "asym"],
                        help="Experimental suite to run (core = primary matrix: R2b, R3-L s=0.8, 1.0 at onsets 2%%, 10%%, 25%%)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds (e.g. 42 or 42 43 44)")
    parser.add_argument("--total_steps", type=int, default=25000, help="Total step budget (reference = 25,000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/cache", help="Persistent dataset cache directory")
    parser.add_argument("--output_dir", type=str, default="./runs/stage1_science", help="Directory to save JSON results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plan = get_experiment_plan(args.suite, args.seeds)

    print("\n" + "=" * 80)
    print(f"STAGE 1 FACTORIAL RUNNER: Suite='{args.suite}' | Total Conditions={len(plan)} | Seeds={args.seeds} | Device={args.device}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    start_all = time.time()
    completed = 0
    skipped = 0

    for idx, cfg in enumerate(plan, 1):
        source = cfg["source"]
        s = cfg["s"]
        asym = cfg["asymmetric_mode"]
        onset = cfg["onset_pct"]
        seed = cfg["seed"]

        # Expected output filename
        out_filename = f"result_{source}_s{s}_onset{onset}_seed{seed}.json"
        if asym:
            out_filename = f"result_{source}_{asym}_onset{onset}_seed{seed}.json"
        out_path = os.path.join(args.output_dir, out_filename)

        if os.path.exists(out_path):
            print(f"[{idx}/{len(plan)}] Skipping already completed: {out_filename}")
            skipped += 1
            continue

        print(f"\n>>> [{idx}/{len(plan)}] Running: Source={source} | s={s} | Asym={asym} | Onset={onset}% | Seed={seed}")
        t0 = time.time()
        try:
            run_science_experiment(
                total_steps=args.total_steps,
                onset_pct=onset,
                source=source,
                s=s,
                asymmetric_mode=asym,
                seed=seed,
                device_name=args.device,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                save_checkpoints=True
            )
            elapsed = time.time() - t0
            completed += 1
            print(f">>> [{idx}/{len(plan)}] Finished {out_filename} in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        except Exception as e:
            print(f"\n[ERROR] Condition {cfg} failed with error: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"STAGE 1 RUNNER COMPLETE: Total={len(plan)} | Finished={completed} | Skipped={skipped} | Elapsed={total_time/60:.1f} minutes")
    print(f"All records saved in {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
