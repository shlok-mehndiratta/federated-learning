"""
Experiment 05: Multi-Client Federated Simulation Pipeline.

Simulates a 10-client federated learning network under Dirichlet non-IID data distribution (alpha = 0.5)
to evaluate model mergeability and head recalibration when clients experience a coordinator outage.

Setup:
- 10 clients partitioned via Dirichlet(0.5) on CIFAR-10.
- Shared trunk pre-training (Rounds 1 to 2).
- Simulated Outage (Rounds 3 to 10): 2 clients disconnect and train locally in isolation,
  while the other 8 clients continue federated training with the server.
- Reconnection at Round 10:
  1. Uninterrupted baseline (all 10 clients continuously synced)
  2. Naive parameter averaging (simple weight merge)
  3. Server-side linear probe recalibration (freeze body, tune classifier on 200 balanced samples)
"""

import argparse
from copy import deepcopy
import json
import os
import random
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_outage.models.resnet import ResNet20LN
from fl_outage.data.cifar import load_cifar10
from fl_outage.data.partition import partition_dataset
from fl_outage.fl.server import average_model_weights
from fl_outage.metrics.barrier import evaluate_loss_and_acc


def set_seed(seed: int = 42):
    # set deterministic random seeds across numpy and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_balanced_calibration_loader(dataset, samples_per_class: int = 20, num_classes: int = 10, seed: int = 42) -> DataLoader:
    # Prototype calibration set: 200 samples (20 per class) sampled directly from the training set for initial feasibility testing.
    # Note: In the v2 experimental harness, server calibration data is formally carved out as a mutually disjoint split before client partitioning.
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    calib_indices = []
    for c in range(num_classes):
        c_idx = np.where(targets == c)[0]
        chosen = rng.choice(c_idx, size=samples_per_class, replace=False)
        calib_indices.extend(chosen)
    subset = Subset(dataset, calib_indices)
    return DataLoader(subset, batch_size=32, shuffle=True)


def train_client_local(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device):
    # local client trainning loop using standard SGD
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()


def recalibrate_linear_head(model: nn.Module, calib_loader: DataLoader, device: torch.device, epochs: int = 5, lr: float = 0.01) -> nn.Module:
    # cure 1 / head recalibration: freeze the entire body and only tune the final fc layer
    calib_model = deepcopy(model).to(device)
    for name, param in calib_model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.SGD(calib_model.fc.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    calib_model.train()
    for _ in range(epochs):
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = calib_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    calib_model.eval()
    return calib_model


def run_simulation(quick_test: bool = False, device_str: str = "cpu", seed: int = 42):
    device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    print(f"Starting 10-client federated simulation on {device} (quick_test={quick_test})...")
    set_seed(seed)

    # 1. Load CIFAR-10 data
    train_dataset, test_dataset = load_cifar10()
    targets = np.array(train_dataset.targets)

    # 10 clients with Dirichlet alpha = 0.5
    num_clients = 10
    # partition dataset across 10 clients with non-iid skew
    partitions = partition_dataset(targets, num_classes=10, s=0.0, alpha=0.5, num_clients_per_group=5, seed=seed)

    # Build client dataloaders
    batch_size = 64 if not quick_test else 32
    client_loaders = []
    for i in range(num_clients):
        idxs = partitions[i]
        if quick_test:
            idxs = idxs[:120]  # small subset for quick smoke test
        subset = Subset(train_dataset, idxs)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(
        Subset(test_dataset, list(range(500 if quick_test else len(test_dataset)))),
        batch_size=128,
        shuffle=False
    )

    # Small balanced server calibration set (200 samples)
    calib_loader = create_balanced_calibration_loader(
        train_dataset,
        samples_per_class=5 if quick_test else 20,
        seed=seed
    )

    # Initialize shared model (ResNet-20 with LayerNorm, 272,474 paramaters)
    global_model = ResNet20LN(num_classes=10).to(device)

    total_rounds = 4 if quick_test else 10
    outage_start_round = 2
    local_epochs = 1 if quick_test else 2
    lr = 0.05

    print(f"Total communication rounds: {total_rounds}, outage from round {outage_start_round + 1} to {total_rounds}")

    # We will simulate:
    # Path A: Uninterrupted baseline (all 10 clients train together)
    # Path B: Outage scenario (clients 8 and 9 disconnect at round 2 and train locally in isolation)
    uninterrupted_model = deepcopy(global_model)
    outage_server_model = deepcopy(global_model)

    # Disconnected client models for outage scenario
    client_models_outage = [deepcopy(global_model) for _ in range(num_clients)]

    # Initial synchronized rounds (Rounds 1 to outage_start_round)
    print("\nRunning initial shared trunk rounds...")
    for r in range(outage_start_round):
        # Sync clients to server
        client_states_unint = []
        client_states_outage = []

        for cid in range(num_clients):
            # uninterrupted path
            c_model = deepcopy(uninterrupted_model)
            train_client_local(c_model, client_loaders[cid], epochs=local_epochs, lr=lr, device=device)
            client_states_unint.append(c_model.state_dict())

            # outage path (still synchronized)
            c_out_model = deepcopy(outage_server_model)
            train_client_local(c_out_model, client_loaders[cid], epochs=local_epochs, lr=lr, device=device)
            client_states_outage.append(c_out_model.state_dict())
            client_models_outage[cid].load_state_dict(c_out_model.state_dict())

        uninterrupted_model.load_state_dict(average_model_weights(client_states_unint))
        outage_server_model.load_state_dict(average_model_weights(client_states_outage))

        _, acc_unint = evaluate_loss_and_acc(uninterrupted_model, test_loader, device=device)
        print(f"  Round {r+1} complete. Server test accuracy: {acc_unint * 100:.2f}%")

    print(f"\nSimulating Server Outage: Clients 8 and 9 disconnected from round {outage_start_round + 1} to {total_rounds}...")
    # Disconnected clients: [8, 9]; Online clients: [0, 1, 2, 3, 4, 5, 6, 7]
    online_cids = list(range(8))
    offline_cids = [8, 9]

    for r in range(outage_start_round, total_rounds):
        # 1. Uninterrupted branch (all 10 continue normal FedAvg)
        client_states_unint = []
        for cid in range(num_clients):
            c_model = deepcopy(uninterrupted_model)
            train_client_local(c_model, client_loaders[cid], epochs=local_epochs, lr=lr, device=device)
            client_states_unint.append(c_model.state_dict())
        uninterrupted_model.load_state_dict(average_model_weights(client_states_unint))

        # 2. Outage branch:
        # Online clients sync with outage_server_model
        online_states = []
        for cid in online_cids:
            c_model = deepcopy(outage_server_model)
            train_client_local(c_model, client_loaders[cid], epochs=local_epochs, lr=lr, device=device)
            online_states.append(c_model.state_dict())
        outage_server_model.load_state_dict(average_model_weights(online_states))

        # Offline clients continue local trainning in isolation (no server aggregation)
        for cid in offline_cids:
            train_client_local(client_models_outage[cid], client_loaders[cid], epochs=local_epochs, lr=lr, device=device)

        _, acc_online = evaluate_loss_and_acc(outage_server_model, test_loader, device=device)
        print(f"  Round {r+1} (Outage ongoing): Online server accuracy = {acc_online * 100:.2f}%")

    # Reconnection at Round 10
    print("\nReconnection phase at Round 10:")
    print("Aggregating server model with isolated client models [8, 9]...")

    # Strategy 1: Naive Parameter Averaging (merge online server model with offline clients)
    # Weight proportional: 8 parts online server (representing 8 clients) + 1 part client 8 + 1 part client 9
    all_states = [outage_server_model.state_dict()] * len(online_cids) + [
        client_models_outage[cid].state_dict() for cid in offline_cids
    ]
    naive_merged_state = average_model_weights(all_states)
    naive_merged_model = deepcopy(global_model)
    naive_merged_model.load_state_dict(naive_merged_state)

    # Strategy 2: Server-side linear probe recalibration
    recalibrated_model = recalibrate_linear_head(naive_merged_model, calib_loader, device=device, epochs=5, lr=0.01)

    # Final Evaluation
    _, unint_acc = evaluate_loss_and_acc(uninterrupted_model, test_loader, device=device)
    _, naive_acc = evaluate_loss_and_acc(naive_merged_model, test_loader, device=device)
    _, recalib_acc = evaluate_loss_and_acc(recalibrated_model, test_loader, device=device)

    print("\n" + "-" * 50)
    print("Multi-Client Simulation Results Summary:")
    print(f"  1. Uninterrupted 10-Client Baseline : {unint_acc * 100:.2f}%")
    print(f"  2. Naive Parameter Averaging (Merged): {naive_acc * 100:.2f}% (Drop: {(unint_acc - naive_acc) * 100:.2f}%)")
    print(f"  3. Server Head Recalibration (Cure 1): {recalib_acc * 100:.2f}% (Gap to Baseline: {(unint_acc - recalib_acc) * 100:.2f}%)")
    print("-" * 50)

    summary = {
        "num_clients": num_clients,
        "dirichlet_alpha": 0.5,
        "total_rounds": total_rounds,
        "outage_start_round": outage_start_round,
        "uninterrupted_acc": unint_acc,
        "naive_merged_acc": naive_acc,
        "recalibrated_acc": recalib_acc,
        "accuracy_drop_naive": unint_acc - naive_acc,
        "gap_after_recalibration": unint_acc - recalib_acc
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Client Federated Simulation Pipeline")
    parser.add_argument("--quick_test", action="store_true", help="Run fast simulation for quick verification")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_simulation(quick_test=args.quick_test, device_str=args.device, seed=args.seed)
