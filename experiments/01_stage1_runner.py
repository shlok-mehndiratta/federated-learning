"""
Stage 1 Factorial Divergence Runner (Matched Design Protocol).
Executes Rung 2b (Disjoint IID control), Rung 3-L (Label Skew), and Asymmetric Divergence.
Saves full trajectory checkpoints every 1% of branch steps to enable D_pair-matched contrasts.

Features:
- Idempotent: checks for existing JSON results in --output_dir and skips completed runs.
- High-Throughput Parallelism: --max_workers allows 4-8 concurrent processes on large GPUs (e.g. 96GB Blackwell).
- Modular suites: --suite core (quick primary matrix), --suite full, --suite r2b, --suite r3l.
- Complete metric logging: 11-point barrier, D_pair, body/head decomposition, CKA, probe gap,
  few-shot head gain, head share, cumulative ZipIt prefixes, logit-offset fidelity.
- Optional Model Checkpointing: --save_models saves model_A and model_B state dicts (.pt) for instant Stage 2 cures.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import os
import subprocess
import sys
import time
from typing import List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def run_single_condition_subprocess(cfg: dict, args) -> Tuple[str, float, int, str]:
    """Runs a single experiment in a dedicated subprocess with its own isolated CUDA context and log file."""
    source = cfg["source"]
    s = cfg["s"]
    asym = cfg["asymmetric_mode"]
    onset = cfg["onset_pct"]
    seed = cfg["seed"]

    out_filename = f"result_{source}_s{s}_onset{onset}_seed{seed}.json"
    if asym:
        out_filename = f"result_{source}_{asym}_onset{onset}_seed{seed}.json"

    log_name = f"log_{source}_s{s}_onset{onset}_seed{seed}.log"
    if asym:
        log_name = f"log_{source}_{asym}_onset{onset}_seed{seed}.log"
    log_path = os.path.join(args.output_dir, log_name)

    cmd = [
        sys.executable, "experiments/science_harness.py",
        "--total_steps", str(args.total_steps),
        "--onset_pct", str(onset),
        "--source", source,
        "--s", str(s),
        "--seed", str(seed),
        "--device", args.device,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ]
    if asym:
        cmd.extend(["--asymmetric_mode", asym])
    if args.save_models:
        cmd.append("--save_models")

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    t0 = time.time()
    with open(log_path, "w") as log_file:
        res = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    return out_filename, elapsed, res.returncode, log_path


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Factorial Divergence Runner (High-Throughput Concurrent Execution)")
    parser.add_argument("--suite", type=str, default="core", choices=["core", "full", "r2b", "r3l", "asym"],
                        help="Experimental suite to run (core = primary matrix: R2b, R3-L s=0.8, 1.0 at onsets 2%%, 10%%, 25%%)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds (e.g. 42 or 42 43 44)")
    parser.add_argument("--total_steps", type=int, default=25000, help="Total step budget (reference = 25,000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/cache", help="Persistent dataset cache directory")
    parser.add_argument("--output_dir", type=str, default="./runs/stage1_science", help="Directory to save JSON results")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of concurrent parallel processes on GPU (e.g. 4 to 6 for 96GB VRAM, 1 for sequential)")
    parser.add_argument("--save_models", action="store_true",
                        help="Save branch model state dicts (.pt) at end of training for instant Stage 2 head cures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plan = get_experiment_plan(args.suite, args.seeds)

    # Filter out already completed runs
    pending_plan = []
    skipped_count = 0
    for cfg in plan:
        source = cfg["source"]
        s = cfg["s"]
        asym = cfg["asymmetric_mode"]
        onset = cfg["onset_pct"]
        seed = cfg["seed"]
        out_filename = f"result_{source}_s{s}_onset{onset}_seed{seed}.json"
        if asym:
            out_filename = f"result_{source}_{asym}_onset{onset}_seed{seed}.json"
        out_path = os.path.join(args.output_dir, out_filename)
        if os.path.exists(out_path):
            skipped_count += 1
        else:
            pending_plan.append(cfg)

    print("\n" + "=" * 80)
    print(f"STAGE 1 FACTORIAL RUNNER: Suite='{args.suite}' | Total Conditions={len(plan)} | Seeds={args.seeds} | Parallel Workers={args.max_workers} | Device={args.device}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Status: {skipped_count} already completed (skipped), {len(pending_plan)} remaining to run.")
    print("=" * 80)

    if not pending_plan:
        print("All requested experiment conditions are already complete! Nothing to do.\n")
        return

    start_all = time.time()
    completed = 0
    failed = 0

    if args.max_workers <= 1:
        # Sequential Execution (in-process live output)
        for idx, cfg in enumerate(pending_plan, 1):
            source = cfg["source"]
            s = cfg["s"]
            asym = cfg["asymmetric_mode"]
            onset = cfg["onset_pct"]
            seed = cfg["seed"]
            out_filename = f"result_{source}_s{s}_onset{onset}_seed{seed}.json"
            if asym:
                out_filename = f"result_{source}_{asym}_onset{onset}_seed{seed}.json"

            print(f"\n>>> [{idx}/{len(pending_plan)}] Running: Source={source} | s={s} | Asym={asym} | Onset={onset}% | Seed={seed}")
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
                    save_checkpoints=True,
                    save_models=args.save_models
                )
                elapsed = time.time() - t0
                completed += 1
                print(f">>> [{idx}/{len(pending_plan)}] Finished {out_filename} in {elapsed:.1f}s ({elapsed/60:.1f}m)")
            except Exception as e:
                print(f"\n[ERROR] Condition {cfg} failed with error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    else:
        # High-Throughput Concurrent Execution
        print(f"\n--> Launching parallel execution pool with {args.max_workers} concurrent workers on {args.device}...")
        futures = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for idx, cfg in enumerate(pending_plan, 1):
                source = cfg["source"]
                s = cfg["s"]
                asym = cfg["asymmetric_mode"]
                onset = cfg["onset_pct"]
                seed = cfg["seed"]
                asym_str = f" | Asym={asym}" if asym else ""
                print(f"  [Scheduled {idx:2d}/{len(pending_plan):2d}] Source={source:<4} | s={s:.1f}{asym_str:<18} | Onset={onset:4.1f}% | Seed={seed}")
                fut = executor.submit(run_single_condition_subprocess, cfg, args)
                futures[fut] = (idx, cfg)

            print(f"\n--> All {len(pending_plan)} jobs queued. Running up to {args.max_workers} simultaneously...\n")
            for fut in as_completed(futures):
                idx, cfg = futures[fut]
                try:
                    out_fname, elapsed, retcode, log_path = fut.result()
                    if retcode == 0:
                        completed += 1
                        print(f"  [DONE {completed:2d}/{len(pending_plan):2d}] {out_fname:<48} in {elapsed:6.1f}s ({elapsed/60:4.1f}m)")
                    else:
                        failed += 1
                        print(f"  [FAIL {failed:2d}/{len(pending_plan):2d}] {out_fname:<48} exited with code {retcode}! (See log: {log_path})")
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] Condition {cfg} threw exception: {e}")

    total_time = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"STAGE 1 RUNNER COMPLETE: Total={len(plan)} | Finished={completed} | Skipped={skipped_count} | Failed={failed} | Elapsed={total_time/60:.1f} minutes")
    print(f"All records saved in {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
