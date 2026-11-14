"""Experiment 03: Milestone 2 - Literature Replication & Pilot Evaluation.

This script executes Milestone M2 / Gate G2:
1. Literature Replication Anchors:
   - Anchor V3b (Zhu et al. reproduction): k=0, s=0.8, full isolation I0, DIFFERENT initializations.
   - Anchor V3a (Our counterpart): k=0, s=0.8, full isolation I0, SHARED initialization.
   - Anchor V4 (Randl reproduction): k>0, s=0.8, dense gossip coupling I2 vs full isolation I0.
2. 1-Seed Pilot Sweep:
   - Grid: s in {0.0, 1.0} x k in {0, 1, 2, 5, 20} rounds x I in {I0, I2}.
   - Seed: 42.

Resource-efficient features:
- Caches global model checkpoints once to avoid redundant retraining.
- Sequential client training guarantees peak VRAM < 300 MB.
- Incremental JSON saving with auto-resume so no work is lost on disconnect.
- Progress logging with flush=True every 5 rounds.
"""

import argparse
from copy import deepcopy
import json
import os
import random
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Subset

from fl_outage.data.cifar import load_cifar10, create_deterministic_loader
from fl_outage.data.partition import partition_dataset
from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.server import FederatedServer
from fl_outage.fl.outage import OutageSimulator
from fl_outage.fl.merge import merge_models
from fl_outage.fl.recovery import run_recovery_evaluation
from fl_outage.metrics.barrier import compute_linear_barrier, evaluate_loss_and_acc
from fl_outage.metrics.distances import compute_normalized_distances
from fl_outage.metrics.curvature import compute_curvature_barrier_estimate


def set_seed(seed: int):
    """Sets deterministic random seed across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_targets(dataset) -> np.ndarray:
    """Safely extracts label targets array from Dataset or Subset."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        # Subset object
        return np.array(dataset.dataset.targets)[dataset.indices]
    else:
        return np.array([y for _, y in dataset])


def evaluate_single_fork(
    model_A: torch.nn.Module,
    model_B: torch.nn.Module,
    model_origin: torch.nn.Module,
    train_set,
    test_set,
    recovery_partitions: Dict[int, List[int]],
    eval_loader,
    target_baseline_acc: float,
    max_recovery_rounds: int,
    lr: float,
    local_steps: int,
    device: torch.device,
    seed: int
) -> Dict[str, any]:
    """Runs standard multi-level evaluation on diverged models A and B."""
    # 1. Geometric metrics
    barrier_res = compute_linear_barrier(model_A, model_B, eval_loader, num_points=11, device=device)
    dist_res = compute_normalized_distances(model_A, model_B, model_anchor=model_origin)
    curvature_est = compute_curvature_barrier_estimate(model_A, model_B, eval_loader, device=device)

    # 2. Individual endpoint accuracy
    _, acc_A = evaluate_loss_and_acc(model_A, eval_loader, device=device)
    _, acc_B = evaluate_loss_and_acc(model_B, eval_loader, device=device)

    # 3. Model merge
    merged_model = merge_models(model_A, model_B)
    _, merged_acc = evaluate_loss_and_acc(merged_model, eval_loader, device=device)

    # 4. Operational recovery loop (Rounds-to-Baseline)
    recovery_res = run_recovery_evaluation(
        merged_model=merged_model,
        dataset=train_set,
        partitions=recovery_partitions,
        eval_loader=eval_loader,
        baseline_target_acc=target_baseline_acc,
        max_recovery_rounds=max_recovery_rounds,
        recovery_tolerance=0.01,
        lr=lr,
        local_steps_per_round=local_steps,
        device=device,
        seed=seed
    )

    return {
        "loss_barrier": barrier_res["barrier"],
        "max_acc_dip": barrier_res["max_acc_dip"],
        "d_pair": dist_res["d_pair"],
        "d_origin": dist_res["d_origin"],
        "curvature_est": curvature_est,
        "acc_group_A": acc_A,
        "acc_group_B": acc_B,
        "merged_acc": merged_acc,
        "recovered": recovery_res["recovered"],
        "rtb": recovery_res["rtb"],
        "final_recovered_acc": recovery_res["final_acc"],
        "recovery_history": recovery_res["history"]
    }


def run_anchors(
    args,
    train_set,
    test_set,
    eval_loader,
    target_baseline_acc: float,
    completed_keys: set,
    all_results: List[Dict],
    out_file: str,
    device: torch.device
):
    """Executes Anchor V3a, V3b, and V4."""
    print("\n" + "=" * 70, flush=True)
    print("EXECUTING WEEK 3 ANCHORS: ZHU ET AL. & RANDL REPLICATION", flush=True)
    print("=" * 70, flush=True)

    targets = get_dataset_targets(train_set)
    group_A_cids = list(range(0, 10))
    group_B_cids = list(range(10, 20))

    # Balanced partitions for recovery evaluation
    recovery_partitions = partition_dataset(
        targets, num_classes=10, s=0.0, alpha=1.0, num_clients_per_group=10, seed=args.seed
    )

    # Heterogeneous partitions for outage (s=0.8)
    outage_partitions = partition_dataset(
        targets, num_classes=10, s=0.8, alpha=1.0, num_clients_per_group=10, seed=args.seed
    )

    # Anchor V3b: Setting with different random inits (k=0, s=0.8, I0)
    v3b_key = ("anchor_v3b", 0, 0.8, "I0", "different_inits")
    if v3b_key in completed_keys:
        print(f"--> [Anchor V3b] Already completed. Skipping.", flush=True)
    else:
        print("\n--> [Anchor V3b] Running Zhu et al. condition (k=0, s=0.8, I0, DIFFERENT inits)...", flush=True)
        set_seed(args.seed)
        model_A_init = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)

        # Different initialization for Model B
        set_seed(args.seed + 999)
        model_B_init = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)

        sim_v3b = OutageSimulator(
            base_model=model_A_init,
            dataset=train_set,
            partitions=outage_partitions,
            group_A_cids=group_A_cids,
            group_B_cids=group_B_cids,
            model_B=model_B_init,
            lr=args.lr,
            local_steps_per_round=args.local_steps,
            device=device
        )
        model_A_v3b, model_B_v3b = sim_v3b.run_outage(
            num_rounds=args.outage_rounds,
            isolation_level="I0",
            seed=args.seed,
            verbose=True
        )

        res_v3b = evaluate_single_fork(
            model_A=model_A_v3b,
            model_B=model_B_v3b,
            model_origin=model_A_init,
            train_set=train_set,
            test_set=test_set,
            recovery_partitions=recovery_partitions,
            eval_loader=eval_loader,
            target_baseline_acc=target_baseline_acc,
            max_recovery_rounds=args.max_recovery_rounds,
            lr=args.lr,
            local_steps=args.local_steps,
            device=device,
            seed=args.seed
        )
        record_v3b = {
            "experiment": "anchor_v3b",
            "onset_round": 0,
            "heterogeneity_s": 0.8,
            "isolation_level": "I0",
            "init_type": "different_inits",
            "seed": args.seed,
            **res_v3b
        }
        all_results.append(record_v3b)
        completed_keys.add(v3b_key)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"    --> V3b (Different Inits): Loss Barrier: {res_v3b['loss_barrier']:.4f} | Merged Acc: {res_v3b['merged_acc']*100:.2f}% | RTB: {res_v3b['rtb']}", flush=True)

    # Anchor V3a: Shared initialization counterpart (k=0, s=0.8, I0)
    v3a_key = ("anchor_v3a", 0, 0.8, "I0", "shared_init")
    if v3a_key in completed_keys:
        print(f"--> [Anchor V3a] Already completed. Skipping.", flush=True)
    else:
        print("\n--> [Anchor V3a] Running Our counterpart (k=0, s=0.8, I0, SHARED init)...", flush=True)
        set_seed(args.seed)
        model_shared = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)

        sim_v3a = OutageSimulator(
            base_model=model_shared,
            dataset=train_set,
            partitions=outage_partitions,
            group_A_cids=group_A_cids,
            group_B_cids=group_B_cids,
            model_B=None,  # Shared init
            lr=args.lr,
            local_steps_per_round=args.local_steps,
            device=device
        )
        model_A_v3a, model_B_v3a = sim_v3a.run_outage(
            num_rounds=args.outage_rounds,
            isolation_level="I0",
            seed=args.seed,
            verbose=True
        )

        res_v3a = evaluate_single_fork(
            model_A=model_A_v3a,
            model_B=model_B_v3a,
            model_origin=model_shared,
            train_set=train_set,
            test_set=test_set,
            recovery_partitions=recovery_partitions,
            eval_loader=eval_loader,
            target_baseline_acc=target_baseline_acc,
            max_recovery_rounds=args.max_recovery_rounds,
            lr=args.lr,
            local_steps=args.local_steps,
            device=device,
            seed=args.seed
        )
        record_v3a = {
            "experiment": "anchor_v3a",
            "onset_round": 0,
            "heterogeneity_s": 0.8,
            "isolation_level": "I0",
            "init_type": "shared_init",
            "seed": args.seed,
            **res_v3a
        }
        all_results.append(record_v3a)
        completed_keys.add(v3a_key)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"    --> V3a (Shared Init): Loss Barrier: {res_v3a['loss_barrier']:.4f} | Merged Acc: {res_v3a['merged_acc']*100:.2f}% | RTB: {res_v3a['rtb']}", flush=True)

    # Anchor V4: Gossip communication (I2) vs complete isolation (I0)
    onset_v4 = 2 if args.quick_test else 5

    # Pre-train shared global model to onset_v4
    print(f"\n--> [Anchor V4] Pre-training shared global model to round {onset_v4}...", flush=True)
    set_seed(args.seed)
    global_model = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)
    server = FederatedServer(
        model=global_model,
        dataset=train_set,
        partitions=recovery_partitions,
        eval_loader=eval_loader,
        lr=args.lr,
        local_steps_per_round=args.local_steps,
        device=device,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints")
    )
    server.run_rounds(num_rounds=onset_v4, seed=args.seed)
    base_v4_state = deepcopy(server.model.state_dict())

    for iso in ["I2", "I0"]:
        v4_key = ("anchor_v4", onset_v4, 0.8, iso, "shared_init")
        if v4_key in completed_keys:
            print(f"--> [Anchor V4 - {iso}] Already completed. Skipping.", flush=True)
            continue

        print(f"\n--> [Anchor V4 - {iso}] Running Outage under {iso} (k={onset_v4}, s=0.8)...", flush=True)
        set_seed(args.seed + (100 if iso == "I2" else 200))
        sim_model = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)
        sim_model.load_state_dict(base_v4_state)

        sim_v4 = OutageSimulator(
            base_model=sim_model,
            dataset=train_set,
            partitions=outage_partitions,
            group_A_cids=group_A_cids,
            group_B_cids=group_B_cids,
            lr=args.lr,
            local_steps_per_round=args.local_steps,
            device=device
        )
        model_A_v4, model_B_v4 = sim_v4.run_outage(
            num_rounds=args.outage_rounds,
            isolation_level=iso,
            seed=args.seed,
            verbose=True
        )

        res_v4 = evaluate_single_fork(
            model_A=model_A_v4,
            model_B=model_B_v4,
            model_origin=sim_model,
            train_set=train_set,
            test_set=test_set,
            recovery_partitions=recovery_partitions,
            eval_loader=eval_loader,
            target_baseline_acc=target_baseline_acc,
            max_recovery_rounds=args.max_recovery_rounds,
            lr=args.lr,
            local_steps=args.local_steps,
            device=device,
            seed=args.seed
        )
        record_v4 = {
            "experiment": "anchor_v4",
            "onset_round": onset_v4,
            "heterogeneity_s": 0.8,
            "isolation_level": iso,
            "init_type": "shared_init",
            "seed": args.seed,
            **res_v4
        }
        all_results.append(record_v4)
        completed_keys.add(v4_key)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"    --> V4 ({iso}): Loss Barrier: {res_v4['loss_barrier']:.4f} | Merged Acc: {res_v4['merged_acc']*100:.2f}% | RTB: {res_v4['rtb']}", flush=True)


def run_pilot(
    args,
    train_set,
    test_set,
    eval_loader,
    target_baseline_acc: float,
    completed_keys: set,
    all_results: List[Dict],
    out_file: str,
    device: torch.device
):
    """Executes the 1-seed pilot sweep using cached trunk checkpoints."""
    print("\n" + "=" * 70, flush=True)
    print("EXECUTING MILESTONE 2: PILOT SWEEP & PRE-REGISTRATION EVALUATION", flush=True)
    print("=" * 70, flush=True)

    targets = get_dataset_targets(train_set)
    group_A_cids = list(range(0, 10))
    group_B_cids = list(range(10, 20))

    if args.quick_test:
        onset_rounds = [0, 2]
        s_levels = [0.0, 1.0]
        isolation_levels = ["I0", "I2"]
    else:
        onset_rounds = [0, 1, 2, 5, 20]
        s_levels = [0.0, 1.0]
        isolation_levels = ["I0", "I2"]

    # 1. Train federated global model once and save checkpoints
    max_onset = max(onset_rounds)
    checkpoint_dir = os.path.join(args.output_dir, "global_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    recovery_partitions = partition_dataset(
        targets, num_classes=10, s=0.0, alpha=1.0, num_clients_per_group=10, seed=args.seed
    )

    print(f"\n--> Training shared global baseline model up to round {max_onset}...", flush=True)
    set_seed(args.seed)
    global_model = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)

    # Save initial model state at round 0
    global_states = {0: deepcopy(global_model.state_dict())}

    if max_onset > 0:
        server = FederatedServer(
            model=global_model,
            dataset=train_set,
            partitions=recovery_partitions,
            eval_loader=eval_loader,
            lr=args.lr,
            local_steps_per_round=args.local_steps,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        for r in range(1, max_onset + 1):
            server.run_rounds(num_rounds=1, seed=args.seed + r * 10)
            if r in onset_rounds:
                global_states[r] = deepcopy(server.model.state_dict())
                print(f"    Saved global checkpoint at Round {r}", flush=True)

    # Pre-generate partitions for each s level
    partition_cache = {}
    for s in s_levels:
        partition_cache[s] = partition_dataset(
            targets, num_classes=10, s=s, alpha=1.0, num_clients_per_group=10, seed=args.seed
        )

    # 2. Sweep across all (s, k, I) cells branching from cached global model states
    total_cells = len(s_levels) * len(onset_rounds) * len(isolation_levels)
    cell_idx = 0

    for s in s_levels:
        for k in onset_rounds:
            for iso in isolation_levels:
                cell_idx += 1
                pilot_key = ("pilot", k, s, iso, "shared_init")

                if pilot_key in completed_keys:
                    print(f"[{cell_idx}/{total_cells}] Cell (s={s:.1f}, k={k}, {iso}) already completed. Skipping.", flush=True)
                    continue

                print(f"\n[{cell_idx}/{total_cells}] Running Pilot Cell: s={s:.1f} | k={k} rounds | {iso}...", flush=True)
                set_seed(args.seed + cell_idx * 100)

                base_model = ResNet20LN(num_classes=10, width_base=args.width_base).to(device)
                base_model.load_state_dict(global_states[k])

                sim = OutageSimulator(
                    base_model=base_model,
                    dataset=train_set,
                    partitions=partition_cache[s],
                    group_A_cids=group_A_cids,
                    group_B_cids=group_B_cids,
                    lr=args.lr,
                    local_steps_per_round=args.local_steps,
                    device=device
                )
                model_A, model_B = sim.run_outage(
                    num_rounds=args.outage_rounds,
                    isolation_level=iso,
                    seed=args.seed + cell_idx * 100,
                    verbose=True
                )

                res = evaluate_single_fork(
                    model_A=model_A,
                    model_B=model_B,
                    model_origin=base_model,
                    train_set=train_set,
                    test_set=test_set,
                    recovery_partitions=recovery_partitions,
                    eval_loader=eval_loader,
                    target_baseline_acc=target_baseline_acc,
                    max_recovery_rounds=args.max_recovery_rounds,
                    lr=args.lr,
                    local_steps=args.local_steps,
                    device=device,
                    seed=args.seed + cell_idx * 100
                )

                record = {
                    "experiment": "pilot",
                    "onset_round": k,
                    "heterogeneity_s": s,
                    "isolation_level": iso,
                    "init_type": "shared_init",
                    "seed": args.seed,
                    **res
                }
                all_results.append(record)
                completed_keys.add(pilot_key)

                with open(out_file, "w") as f:
                    json.dump(all_results, f, indent=2)

                print(f"--> [Cell Result] Barrier: {res['loss_barrier']:.4f} | D_pair: {res['d_pair']:.4f} | Merged Acc: {res['merged_acc']*100:.2f}% | RTB: {res['rtb']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Milestone 2: Literature Replication & Pilot Evaluation")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "anchors", "pilot"])
    parser.add_argument("--data_root", type=str, default="./data/cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./runs/pilot_evaluation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--outage_rounds", type=int, default=20)
    parser.add_argument("--max_recovery_rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width_base", type=int, default=16)
    parser.add_argument("--quick_test", action="store_true", help="Run 15s quick CPU smoke test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Starting Milestone 2 Runner on device: {device} | Mode: {args.mode}")

    # Load CIFAR-10
    train_set, test_set = load_cifar10(root=args.data_root, download=True)

    if args.quick_test:
        print("--- RUNNING IN QUICK TEST MODE (Miniature dataset & widths) ---")
        train_set = Subset(train_set, list(range(400)))
        test_set = Subset(test_set, list(range(200)))
        args.width_base = 4
        args.outage_rounds = 2
        args.max_recovery_rounds = 3
        args.local_steps = 2
        args.batch_size = 32

    eval_loader = create_deterministic_loader(test_set, batch_size=256, shuffle=False)

    # Estimate unbroken baseline accuracy target (provisional target for RTB tolerance check)
    target_baseline_acc = 0.35 if args.quick_test else 0.28

    out_file = os.path.join(args.output_dir, "pilot_results.json")
    # Support backward compatibility with existing week3_results.json
    legacy_file = os.path.join(args.output_dir, "week3_results.json")
    if not os.path.exists(out_file) and os.path.exists(legacy_file):
        out_file = legacy_file

    all_results = []
    completed_keys = set()

    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                all_results = json.load(f)
            for r in all_results:
                key = (
                    r.get("experiment", "unknown"),
                    r.get("onset_round", 0),
                    r.get("heterogeneity_s", 0.0),
                    r.get("isolation_level", "I0"),
                    r.get("init_type", "shared_init")
                )
                completed_keys.add(key)
            print(f"Loaded {len(all_results)} existing results from {out_file}. Resuming cleanly...")
        except Exception as e:
            print(f"Warning: Could not parse existing results ({e}). Starting fresh.")

    if args.mode in ("all", "anchors"):
        run_anchors(
            args, train_set, test_set, eval_loader, target_baseline_acc,
            completed_keys, all_results, out_file, device
        )

    if args.mode in ("all", "pilot"):
        run_pilot(
            args, train_set, test_set, eval_loader, target_baseline_acc,
            completed_keys, all_results, out_file, device
        )

    print("\n" + "=" * 70, flush=True)
    print(f"MILESTONE 2 RUN COMPLETED SUCCESSFULLY! Results saved to: {out_file}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
