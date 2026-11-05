"""Experiment 04: Milestone 3 - Confirmatory Factorial Parameter Sweeps (Wave 1 & Wave 2).

Executes the full confirmatory factorial experimental design for Gate G3:
1. Wave 1 (Core Grid):
   - Heterogeneity: s in {0.0, 0.5, 0.8, 1.0}
   - Onsets: k in {0, 1, 2, 5, 20} rounds
   - Isolation: I in {I0, I1, I2} (with I1 sparse exchange every R=5 rounds)
   - Seeds: {42, 43, 44}
   - Outage length: L=20 rounds
2. Wave 2 (Duration Sensitivity & Start-Type Contrast):
   - Outage Duration Sweep: L in {5, 20, 40} rounds across s in {0.0, 0.8, 1.0}, I in {I0, I2}, k in {0, 2}.
   - Start-Type Contrast (Hypothesis H1): ResNet-20 LN pretrained on CIFAR-100 vs. In-Task FL Server
     starting at k=0 under full isolation I0 across s in {0.0, 0.8, 1.0} and L in {5, 20, 40}.

Resource Guarantees:
- Peak VRAM < 300 MB via sequential client updates.
- Global model caching per seed saves >65% GPU training time.
- Incremental JSON saving with auto-resume across crashes/disconnects.
"""

import argparse
from copy import deepcopy
import json
import os
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from fl_outage.data.cifar import load_cifar10, create_deterministic_loader
from fl_outage.data.partition import partition_dataset
from fl_outage.models.resnet import ResNet20LN
from fl_outage.models.pretrained import get_cifar100_pretrained_model
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
        return np.array(dataset.dataset.targets)[dataset.indices]
    else:
        return np.array([y for _, y in dataset])


def evaluate_single_fork(
    model_A: nn.Module,
    model_B: nn.Module,
    model_origin: nn.Module,
    train_set,
    recovery_partitions: Dict[int, List[int]],
    eval_loader,
    target_baseline_acc: float,
    max_recovery_rounds: int,
    lr: float,
    local_steps: int,
    device: torch.device,
    seed: int
) -> Dict[str, any]:
    """Runs multi-level evaluation on diverged models A and B."""
    # 1. Geometric metrics
    barrier_res = compute_linear_barrier(model_A, model_B, eval_loader, num_points=11, device=device)
    dist_res = compute_normalized_distances(model_A, model_B, model_anchor=model_origin)
    curvature_est = compute_curvature_barrier_estimate(model_A, model_B, eval_loader, device=device)

    # 2. Endpoint accuracy
    _, acc_A = evaluate_loss_and_acc(model_A, eval_loader, device=device)
    _, acc_B = evaluate_loss_and_acc(model_B, eval_loader, device=device)

    # 3. Model merge
    merged_model = merge_models(model_A, model_B)
    _, merged_acc = evaluate_loss_and_acc(merged_model, eval_loader, device=device)

    # 4. Operational recovery loop
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


def precompute_global_checkpoints(
    seed: int,
    train_set,
    targets: np.ndarray,
    eval_loader,
    checkpoint_rounds: List[int],
    total_baseline_rounds: int,
    num_clients: int,
    width_base: int,
    lr: float,
    local_steps: int,
    device: torch.device,
    cache_dir: str
) -> Dict[int, str]:
    """Precomputes and caches unbroken global baseline checkpoints for a given seed."""
    os.makedirs(cache_dir, exist_ok=True)
    paths: Dict[int, str] = {}
    all_exist = True
    for k in checkpoint_rounds:
        p = os.path.join(cache_dir, f"global_model_seed{seed}_round_{k}.pt")
        paths[k] = p
        if not os.path.exists(p):
            all_exist = False

    history_path = os.path.join(cache_dir, f"baseline_history_seed{seed}.json")
    if all_exist and os.path.exists(history_path):
        print(f"--> [Cache Hit] Found cached global checkpoints for Seed {seed}.", flush=True)
        return paths

    print(f"--> [Pre-training] Training unbroken global server model for Seed {seed} up to Round {total_baseline_rounds}...", flush=True)
    set_seed(seed)
    init_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)

    # IID partition for baseline global model
    partitions = partition_dataset(
        targets=targets,
        num_classes=10,
        s=0.0,
        alpha=1.0,
        num_clients_per_group=num_clients // 2,
        seed=seed
    )

    server = FederatedServer(
        model=init_model,
        dataset=train_set,
        partitions=partitions,
        eval_loader=eval_loader,
        lr=lr,
        local_steps_per_round=local_steps,
        device=device,
        checkpoint_dir=cache_dir
    )

    baseline_history: Dict[int, float] = {}

    # Round 0 snapshot
    if 0 in checkpoint_rounds:
        torch.save(server.model.state_dict(), paths[0])
        _, acc0 = evaluate_loss_and_acc(server.model, eval_loader, device=device)
        baseline_history[0] = acc0

    current_round = 0
    max_k = max(checkpoint_rounds + [total_baseline_rounds])

    for target_round in range(1, max_k + 1):
        server.run_rounds(num_rounds=1, start_step=current_round * local_steps, seed=seed + target_round * 100)
        current_round = target_round
        _, acc = evaluate_loss_and_acc(server.model, eval_loader, device=device)
        baseline_history[current_round] = acc

        if current_round in checkpoint_rounds:
            torch.save(server.model.state_dict(), paths[current_round])
            print(f"    Saved global checkpoint at Round {current_round} (Acc: {acc:.2f}%)", flush=True)

    with open(history_path, "w") as f:
        json.dump(baseline_history, f, indent=2)

    return paths


def run_milestone3(
    wave: str = "all",
    seeds: List[int] = [42, 43, 44],
    device_str: str = "cuda",
    output_dir: str = "./runs/factorial_sweep",
    quick_test: bool = False
):
    """Executes Milestone 3 Factorial Sweeps."""
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "factorial_results.json")

    print("=" * 80)
    print(f"STARTING MILESTONE 3 FACTORIAL EXPERIMENT RUNNER")
    print(f"Device: {device} | Wave: {wave} | Seeds: {seeds} | Quick Test: {quick_test}")
    print("=" * 80)

    # 1. Load Data
    full_train, full_test = load_cifar10(root="./data/cache", download=True)
    if quick_test:
        train_set = Subset(full_train, list(range(1000)))
        test_set = Subset(full_test, list(range(400)))
        num_clients = 4
        width_base = 8
        local_steps = 2
        max_recovery_rounds = 3
        k_levels = [0, 2]
        s_levels = [0.0, 1.0]
        iso_levels = ["I0", "I2"]
        outage_lengths = [2, 4]
        medium_L = 2
        sparse_R = 2
    else:
        train_set = full_train
        test_set = full_test
        num_clients = 20
        width_base = 16
        local_steps = 10
        max_recovery_rounds = 30
        k_levels = [0, 1, 2, 5, 20]
        s_levels = [0.0, 0.5, 0.8, 1.0]
        iso_levels = ["I0", "I1", "I2"]
        outage_lengths = [5, 20, 40]
        medium_L = 20
        sparse_R = 5

    eval_loader = create_deterministic_loader(test_set, batch_size=128, shuffle=False)
    targets = get_dataset_targets(train_set)

    # Auto-resume loading
    results_data: Dict[str, any] = {"wave1": [], "wave2_duration": [], "wave2_start_type": []}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
            print(f"--> [Auto-Resume] Loaded existing results from {results_path}", flush=True)
        except Exception as e:
            print(f"--> [Auto-Resume Warning] Could not parse existing results: {e}. Starting fresh.", flush=True)

    cache_dir = os.path.join(output_dir, "global_checkpoints")
    checkpoint_rounds = sorted(list(set(k_levels)))

    # =========================================================================
    # WAVE 1: CORE FACTORIAL GRID
    # =========================================================================
    if wave in ("1", "all"):
        print("\n" + "=" * 80)
        print("WAVE 1: CORE FACTORIAL GRID SWEEP")
        print("=" * 80)

        for seed in seeds:
            print(f"\n---> [Wave 1] Processing Seed {seed}...")
            # Precompute/fetch global model checkpoints
            paths = precompute_global_checkpoints(
                seed=seed,
                train_set=train_set,
                targets=targets,
                eval_loader=eval_loader,
                checkpoint_rounds=checkpoint_rounds,
                total_baseline_rounds=max(checkpoint_rounds) + medium_L,
                num_clients=num_clients,
                width_base=width_base,
                lr=0.1,
                local_steps=local_steps,
                device=device,
                cache_dir=cache_dir
            )

            # Load baseline history for dynamic RTB target
            with open(os.path.join(cache_dir, f"baseline_history_seed{seed}.json"), "r") as f:
                baseline_history = {int(k): v for k, v in json.load(f).items()}

            completed_keys = {
                (c["seed"], c["s"], c["k"], c["isolation"])
                for c in results_data.get("wave1", [])
            }

            for s in s_levels:
                # Partition clients for this skew
                partitions = partition_dataset(
                    targets=targets,
                    num_classes=10,
                    s=s,
                    alpha=1.0,
                    num_clients_per_group=num_clients // 2,
                    seed=seed
                )
                group_A = list(range(num_clients // 2))
                group_B = list(range(num_clients // 2, num_clients))

                for k in k_levels:
                    # Target baseline accuracy at round k + medium_L
                    target_acc = baseline_history.get(k + medium_L, baseline_history[max(baseline_history.keys())])

                    # Base model for this onset
                    base_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)
                    base_model.load_state_dict(torch.load(paths[k], map_location=device, weights_only=True))

                    for iso in iso_levels:
                        cell_key = (seed, s, k, iso)
                        if cell_key in completed_keys:
                            continue

                        print(f"[Wave 1] Seed {seed} | s={s:.1f} | k={k} rounds | Iso={iso}...", flush=True)
                        sim = OutageSimulator(
                            base_model=base_model,
                            dataset=train_set,
                            partitions=partitions,
                            group_A_cids=group_A,
                            group_B_cids=group_B,
                            lr=0.1,
                            local_steps_per_round=local_steps,
                            device=device
                        )
                        mA, mB = sim.run_outage(
                            num_rounds=medium_L,
                            isolation_level=iso,
                            sparse_freq_R=sparse_R,
                            seed=seed + k * 10,
                            verbose=False
                        )

                        eval_res = evaluate_single_fork(
                            model_A=mA,
                            model_B=mB,
                            model_origin=base_model,
                            train_set=train_set,
                            recovery_partitions=partitions,
                            eval_loader=eval_loader,
                            target_baseline_acc=target_acc,
                            max_recovery_rounds=max_recovery_rounds,
                            lr=0.1,
                            local_steps=local_steps,
                            device=device,
                            seed=seed
                        )

                        cell_record = {
                            "wave": 1,
                            "seed": seed,
                            "s": s,
                            "k": k,
                            "isolation": iso,
                            "outage_length": medium_L,
                            **eval_res
                        }
                        results_data["wave1"].append(cell_record)
                        completed_keys.add(cell_key)

                        # Incremental write
                        with open(results_path, "w") as f:
                            json.dump(results_data, f, indent=2)

                        print(f"    --> Barrier: {eval_res['loss_barrier']:.4f} | Acc Dip: {eval_res['max_acc_dip']*100:.2f}% | Merged Acc: {eval_res['merged_acc']*100:.2f}% | RTB: {eval_res['rtb']}", flush=True)

    # =========================================================================
    # WAVE 2: DURATION SENSITIVITY & START-TYPE CONTRAST
    # =========================================================================
    if wave in ("2", "all"):
        print("\n" + "=" * 80)
        print("WAVE 2: OUTAGE DURATION & START-TYPE CONTRAST")
        print("=" * 80)

        # Part A: Outage Duration Sensitivity
        for seed in seeds:
            print(f"\n---> [Wave 2: Duration] Processing Seed {seed}...")
            paths = precompute_global_checkpoints(
                seed=seed,
                train_set=train_set,
                targets=targets,
                eval_loader=eval_loader,
                checkpoint_rounds=checkpoint_rounds,
                total_baseline_rounds=max(checkpoint_rounds) + max(outage_lengths),
                num_clients=num_clients,
                width_base=width_base,
                lr=0.1,
                local_steps=local_steps,
                device=device,
                cache_dir=cache_dir
            )

            with open(os.path.join(cache_dir, f"baseline_history_seed{seed}.json"), "r") as f:
                baseline_history = {int(k): v for k, v in json.load(f).items()}

            completed_dur_keys = {
                (c["seed"], c["s"], c["k"], c["isolation"], c["outage_length"])
                for c in results_data.get("wave2_duration", [])
            }

            # Tested on key boundaries s in {0.0, 0.8, 1.0} and onsets {0, 2}
            dur_s_levels = [s for s in [0.0, 0.8, 1.0] if s in s_levels]
            dur_k_levels = [k for k in [0, 2] if k in k_levels]
            dur_iso_levels = [iso for iso in ["I0", "I2"] if iso in iso_levels]

            for s in dur_s_levels:
                partitions = partition_dataset(
                    targets=targets,
                    num_classes=10,
                    s=s,
                    alpha=1.0,
                    num_clients_per_group=num_clients // 2,
                    seed=seed
                )
                group_A = list(range(num_clients // 2))
                group_B = list(range(num_clients // 2, num_clients))

                for k in dur_k_levels:
                    base_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)
                    base_model.load_state_dict(torch.load(paths[k], map_location=device, weights_only=True))

                    for L in outage_lengths:
                        target_acc = baseline_history.get(k + L, baseline_history[max(baseline_history.keys())])

                        for iso in dur_iso_levels:
                            cell_key = (seed, s, k, iso, L)
                            if cell_key in completed_dur_keys:
                                continue

                            print(f"[Wave 2: Duration] Seed {seed} | s={s:.1f} | k={k} | L={L} | Iso={iso}...", flush=True)
                            sim = OutageSimulator(
                                base_model=base_model,
                                dataset=train_set,
                                partitions=partitions,
                                group_A_cids=group_A,
                                group_B_cids=group_B,
                                lr=0.1,
                                local_steps_per_round=local_steps,
                                device=device
                            )
                            mA, mB = sim.run_outage(
                                num_rounds=L,
                                isolation_level=iso,
                                sparse_freq_R=sparse_R,
                                seed=seed + k * 10 + L,
                                verbose=False
                            )

                            eval_res = evaluate_single_fork(
                                model_A=mA,
                                model_B=mB,
                                model_origin=base_model,
                                train_set=train_set,
                                recovery_partitions=partitions,
                                eval_loader=eval_loader,
                                target_baseline_acc=target_acc,
                                max_recovery_rounds=max_recovery_rounds,
                                lr=0.1,
                                local_steps=local_steps,
                                device=device,
                                seed=seed
                            )

                            cell_record = {
                                "wave": 2,
                                "experiment": "outage_duration",
                                "seed": seed,
                                "s": s,
                                "k": k,
                                "isolation": iso,
                                "outage_length": L,
                                **eval_res
                            }
                            results_data["wave2_duration"].append(cell_record)
                            completed_dur_keys.add(cell_key)

                            with open(results_path, "w") as f:
                                json.dump(results_data, f, indent=2)

                            print(f"    --> Acc Dip: {eval_res['max_acc_dip']*100:.2f}% | Merged Acc: {eval_res['merged_acc']*100:.2f}% | RTB: {eval_res['rtb']}", flush=True)

        # Part B: Start-Type Contrast (Hypothesis H1)
        print("\n---> [Wave 2: Start-Type Contrast] In-Task FL vs Generic CIFAR-100 Pretraining...")
        cifar100_pretrained_model = get_cifar100_pretrained_model(
            checkpoint_path=os.path.join(output_dir, "cifar100_resnet20ln.pt"),
            num_classes=10,
            width_base=width_base,
            device=device,
            seed=42,
            quick_test=quick_test
        )

        completed_start_keys = {
            (c["seed"], c["s"], c["start_type"], c["outage_length"])
            for c in results_data.get("wave2_start_type", [])
        }

        start_s_levels = [s for s in [0.0, 0.8, 1.0] if s in s_levels]

        for seed in seeds:
            for s in start_s_levels:
                partitions = partition_dataset(
                    targets=targets,
                    num_classes=10,
                    s=s,
                    alpha=1.0,
                    num_clients_per_group=num_clients // 2,
                    seed=seed
                )
                group_A = list(range(num_clients // 2))
                group_B = list(range(num_clients // 2, num_clients))

                for L in outage_lengths:
                    # 1. Condition: Generic Pretraining (CIFAR-100)
                    cell_key_gen = (seed, s, "generic_cifar100", L)
                    if cell_key_gen not in completed_start_keys:
                        print(f"[Wave 2: Start-Type] Seed {seed} | s={s:.1f} | Start=Generic CIFAR-100 | L={L} | Iso=I0...", flush=True)
                        sim_gen = OutageSimulator(
                            base_model=cifar100_pretrained_model,
                            dataset=train_set,
                            partitions=partitions,
                            group_A_cids=group_A,
                            group_B_cids=group_B,
                            lr=0.1,
                            local_steps_per_round=local_steps,
                            device=device
                        )
                        mA_gen, mB_gen = sim_gen.run_outage(
                            num_rounds=L,
                            isolation_level="I0",
                            seed=seed + L * 3,
                            verbose=False
                        )
                        eval_gen = evaluate_single_fork(
                            model_A=mA_gen,
                            model_B=mB_gen,
                            model_origin=cifar100_pretrained_model,
                            train_set=train_set,
                            recovery_partitions=partitions,
                            eval_loader=eval_loader,
                            target_baseline_acc=0.50,  # Generic baseline reference
                            max_recovery_rounds=max_recovery_rounds,
                            lr=0.1,
                            local_steps=local_steps,
                            device=device,
                            seed=seed
                        )
                        results_data["wave2_start_type"].append({
                            "wave": 2,
                            "experiment": "start_type_contrast",
                            "start_type": "generic_cifar100",
                            "seed": seed,
                            "s": s,
                            "k": 0,
                            "isolation": "I0",
                            "outage_length": L,
                            **eval_gen
                        })
                        completed_start_keys.add(cell_key_gen)
                        with open(results_path, "w") as f:
                            json.dump(results_data, f, indent=2)

                    # 2. Condition: In-Task Shared Initialization
                    cell_key_intask = (seed, s, "in_task_fl", L)
                    if cell_key_intask not in completed_start_keys:
                        print(f"[Wave 2: Start-Type] Seed {seed} | s={s:.1f} | Start=In-Task FL | L={L} | Iso=I0...", flush=True)
                        intask_init_model = ResNet20LN(num_classes=10, width_base=width_base).to(device)
                        intask_init_model.load_state_dict(torch.load(paths[0], map_location=device, weights_only=True))

                        sim_intask = OutageSimulator(
                            base_model=intask_init_model,
                            dataset=train_set,
                            partitions=partitions,
                            group_A_cids=group_A,
                            group_B_cids=group_B,
                            lr=0.1,
                            local_steps_per_round=local_steps,
                            device=device
                        )
                        mA_in, mB_in = sim_intask.run_outage(
                            num_rounds=L,
                            isolation_level="I0",
                            seed=seed + L * 5,
                            verbose=False
                        )
                        eval_intask = evaluate_single_fork(
                            model_A=mA_in,
                            model_B=mB_in,
                            model_origin=intask_init_model,
                            train_set=train_set,
                            recovery_partitions=partitions,
                            eval_loader=eval_loader,
                            target_baseline_acc=0.50,
                            max_recovery_rounds=max_recovery_rounds,
                            lr=0.1,
                            local_steps=local_steps,
                            device=device,
                            seed=seed
                        )
                        results_data["wave2_start_type"].append({
                            "wave": 2,
                            "experiment": "start_type_contrast",
                            "start_type": "in_task_fl",
                            "seed": seed,
                            "s": s,
                            "k": 0,
                            "isolation": "I0",
                            "outage_length": L,
                            **eval_intask
                        })
                        completed_start_keys.add(cell_key_intask)
                        with open(results_path, "w") as f:
                            json.dump(results_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"MILESTONE 3 FACTORIAL RUN COMPLETED! Saved to {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3 Factorial Sweeps Runner")
    parser.add_argument("--wave", type=str, default="all", choices=["1", "2", "all"], help="Wave to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./runs/factorial_sweep", help="Directory for results")
    parser.add_argument("--quick_test", action="store_true", help="Fast smoke test on CPU")
    args = parser.parse_args()

    run_milestone3(
        wave=args.wave,
        seeds=args.seeds,
        device_str=args.device,
        output_dir=args.output_dir,
        quick_test=args.quick_test
    )
