"""Experiment 02: Milestone 1 - Federated Learning Stability Boundary Evaluation (s=0.0).

Applies the perturb-and-fork evaluation to a 20-client Federated Learning setup under IID data (s=0.0):
1. Trains global FedAvg model with plain SGD (constant lr=0.1, no momentum).
2. At specified onset step k, forks the global model into Replica 1 and Replica 2.
3. Applies a controlled batch-gradient perturbation to Replica 2.
4. Trains both replicas for remaining rounds with identical client sampling and batch ordering.
5. Evaluates Sharma et al. 11-point loss barrier and parameter distance D_pair.
6. Pinpoints the empirical boundary of the chaotic/unstable phase in FL gradient steps.
"""

import argparse
from copy import deepcopy
import json
import math
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_outage.data.cifar import load_cifar10, create_deterministic_loader
from fl_outage.data.partition import partition_dataset
from fl_outage.models.resnet import ResNet20LN
from fl_outage.fl.client import FederatedClient
from fl_outage.fl.server import average_model_weights
from fl_outage.metrics.barrier import compute_linear_barrier, evaluate_loss_and_acc
from fl_outage.metrics.distances import compute_normalized_distances

try:
    import wandb
except ImportError:
    wandb = None


def apply_batch_perturbation(
    model: nn.Module,
    sample_loader: DataLoader,
    device: torch.device,
    scale: float = 0.01
) -> float:
    """Applies a normalized gradient perturbation from a sample batch:
        W <- W + scale * (grad / ||grad||)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(sample_loader))
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad.data ** 2).item()
    total_norm = math.sqrt(total_norm_sq)

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None and total_norm > 0:
                p.data.add_(p.grad.data, alpha=(scale / total_norm))
    model.zero_grad()
    return total_norm


def run_v2_fork(
    onset_round: int,
    total_rounds: int,
    seed: int,
    device: torch.device,
    train_set,
    test_set,
    partitions: Dict[int, List[int]],
    local_steps: int = 10,
    batch_size: int = 128,
    lr: float = 0.1,
    perturb_scale: float = 0.01,
    num_eval_points: int = 11,
    use_wandb: bool = False
) -> Dict[str, any]:
    """Runs a single federated perturb-and-fork experiment at s=0 for total_rounds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize shared model and clients
    global_model = ResNet20LN(num_classes=10, width_base=16).to(device)
    clients = [
        FederatedClient(cid, train_set, partitions[cid], batch_size=batch_size, lr=lr, device=device)
        for cid in sorted(partitions.keys())
    ]
    test_loader = create_deterministic_loader(test_set, batch_size=256, shuffle=False)

    round_idx = 0
    print(f"\n[Seed {seed}] Running federated training to onset round {onset_round}/{total_rounds}...", flush=True)

    # Phase 1: Train global model until onset_round
    while round_idx < onset_round:
        round_seed = seed + round_idx * 1000
        client_states = []
        base_state = deepcopy(global_model.state_dict())

        for client in clients:
            global_model.load_state_dict(base_state)
            client.train_steps(global_model, num_steps=local_steps, seed=round_seed + client.client_id)
            client_states.append(deepcopy(global_model.state_dict()))

        avg_state = average_model_weights(client_states)
        global_model.load_state_dict(avg_state)
        round_idx += 1

        if round_idx % 5 == 0 or round_idx >= onset_round:
            print(f"  [Global] Round {round_idx}/{onset_round}", flush=True)

    # Phase 2: Fork into Replica 1 and Replica 2
    model_rep1 = deepcopy(global_model)
    model_rep2 = deepcopy(global_model)

    # Perturb Replica 2
    perturb_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed + 9999)
    apply_batch_perturbation(model_rep2, perturb_loader, device=device, scale=perturb_scale)

    print(f"[Seed {seed}] Training both FL replicas for remaining rounds ({total_rounds - round_idx})...", flush=True)

    # Phase 3: Train both replicas for remaining rounds with identical data
    while round_idx < total_rounds:
        round_seed = seed + round_idx * 1000

        # Step Replica 1
        states_1 = []
        base_1 = deepcopy(model_rep1.state_dict())
        for client in clients:
            model_rep1.load_state_dict(base_1)
            client.train_steps(model_rep1, num_steps=local_steps, seed=round_seed + client.client_id)
            states_1.append(deepcopy(model_rep1.state_dict()))
        model_rep1.load_state_dict(average_model_weights(states_1))

        # Step Replica 2 (identical client and batch seed)
        states_2 = []
        base_2 = deepcopy(model_rep2.state_dict())
        for client in clients:
            model_rep2.load_state_dict(base_2)
            client.train_steps(model_rep2, num_steps=local_steps, seed=round_seed + client.client_id)
            states_2.append(deepcopy(model_rep2.state_dict()))
        model_rep2.load_state_dict(average_model_weights(states_2))

        round_idx += 1

        if round_idx % 5 == 0 or round_idx >= total_rounds:
            pct = (round_idx / total_rounds) * 100.0
            print(f"  [Replicas] Round {round_idx}/{total_rounds} ({pct:.1f}%)", flush=True)

    # Phase 4: Measurement
    print(f"[Seed {seed}] Measuring 11-point barrier between replicas...", flush=True)
    barrier_res = compute_linear_barrier(model_rep1, model_rep2, test_loader, num_points=num_eval_points, device=device)
    dist_res = compute_normalized_distances(model_rep1, model_rep2, model_anchor=global_model)
    _, acc1 = evaluate_loss_and_acc(model_rep1, test_loader, device=device)
    _, acc2 = evaluate_loss_and_acc(model_rep2, test_loader, device=device)

    result = {
        "onset_round": onset_round,
        "onset_pct": float(onset_round) / float(total_rounds) * 100.0,
        "seed": seed,
        "loss_barrier": barrier_res["barrier"],
        "acc_dip": barrier_res["max_acc_dip"],
        "d_pair": dist_res["d_pair"],
        "d_origin": dist_res["d_origin"],
        "acc_rep1": acc1,
        "acc_rep2": acc2,
        "merged_acc": barrier_res["merged_acc"],
    }

    print(f"--> [FL Stability] Onset {result['onset_pct']:.1f}% (Round {onset_round}) | Loss Barrier: {result['loss_barrier']:.4f} | D_pair: {result['d_pair']:.4f} | Merged Acc: {result['merged_acc']*100:.2f}%", flush=True)

    if use_wandb and wandb is not None:
        wandb.log({
            "fl_stability/onset_pct": result["onset_pct"],
            "fl_stability/loss_barrier": result["loss_barrier"],
            "fl_stability/d_pair": result["d_pair"],
            "fl_stability/merged_acc": result["merged_acc"],
            "fl_stability/seed": seed
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Milestone 1: Federated Stability Boundary Evaluation (s=0.0)")
    parser.add_argument("--rounds", type=int, default=100, help="Total global FL communication rounds (default 100 = 50 CIFAR-10 epochs)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--local_steps", type=int, default=10, help="Local gradient steps per client per round")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--perturb_scale", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./runs/federated_stability")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fl-federated-stability")
    parser.add_argument("--toy_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running Federated Stability Evaluation on device: {device}")

    train_set, test_set = load_cifar10(root=args.data_root, download=True)
    targets = np.array(train_set.targets)

    # IID partition (s=0.0) across 20 clients
    partitions = partition_dataset(targets, num_classes=10, s=0.0, alpha=1.0, num_clients_per_group=10, seed=42)

    total_rounds = args.rounds

    if args.toy_run:
        print("--- TOY RUN MODE (Smoke test) ---")
        total_rounds = 6
        onset_rounds = [0, 2, 4]
        args.seeds = [1]
        args.local_steps = 2
    else:
        # Onset percentages across total rounds: 0%, 1%, 2%, 5%, 10%, 20%, 50%
        onset_pcts = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        onset_rounds = [int(p * total_rounds) for p in onset_pcts]

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    out_file = os.path.join(args.output_dir, "federated_results.json")
    # Support backward compatibility with existing v2_results.json
    legacy_file = os.path.join(args.output_dir, "v2_results.json")
    if not os.path.exists(out_file) and os.path.exists(legacy_file):
        out_file = legacy_file
    all_results = []
    completed_keys = set()
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                all_results = json.load(f)
            for r in all_results:
                completed_keys.add((r.get("onset_round", r.get("onset_step")), r["seed"]))
            print(f"Loaded {len(all_results)} existing results from {out_file}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not read existing results ({e}). Starting fresh.")

    for onset_r in onset_rounds:
        for seed in args.seeds:
            if (onset_r, seed) in completed_keys:
                print(f"--> Skipping Onset Round {onset_r} (Seed {seed}) - already completed.")
                continue

            res = run_v2_fork(
                onset_round=onset_r,
                total_rounds=total_rounds,
                seed=seed,
                device=device,
                train_set=train_set,
                test_set=test_set,
                partitions=partitions,
                local_steps=args.local_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                perturb_scale=args.perturb_scale,
                use_wandb=args.use_wandb
            )
            all_results.append(res)
            completed_keys.add((onset_r, seed))

            # Incremental save so no progress is lost on disconnect
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll V2 runs complete! Results saved to {out_file}")


if __name__ == "__main__":
    main()
