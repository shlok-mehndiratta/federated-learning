"""Pretrained model utilities for Start-Type Contrast (Hypothesis H1).

Supports loading or pretraining ResNet-20 LN on CIFAR-100, and adapting
the feature backbone to CIFAR-10 by replacing the final linear layer.
"""

from copy import deepcopy
import os
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_outage.data.cifar import load_cifar100, create_deterministic_loader
from fl_outage.models.resnet import ResNet20LN


def pretrain_resnet20_cifar100(
    output_path: str = "./checkpoints/cifar100_resnet20ln.pt",
    epochs: int = 20,
    lr: float = 0.1,
    batch_size: int = 128,
    width_base: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,

    custom_dataset: Optional[torch.utils.data.Dataset] = None,
    quick_test: bool = False
) -> nn.Module:

    """Pre-trains ResNet20LN on CIFAR-100 and saves the checkpoint."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.manual_seed(seed)

    if custom_dataset is not None:
        train_set = custom_dataset
    elif quick_test:
        from torchvision.datasets import FakeData
        import torchvision.transforms as transforms
        train_set = FakeData(size=128, image_size=(3, 32, 32), num_classes=100, transform=transforms.ToTensor())
        epochs = 1
    else:
        train_set, _ = load_cifar100(root="./data/cache", download=True)

    train_loader = create_deterministic_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed)

    model = ResNet20LN(num_classes=100, width_base=width_base).to(device)


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), output_path)
    return model


def get_cifar100_pretrained_model(
    checkpoint_path: str = "./checkpoints/cifar100_resnet20ln.pt",
    num_classes: int = 10,
    width_base: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    quick_test: bool = False
) -> ResNet20LN:
    """Loads a CIFAR-100 pretrained ResNet-20 LN and adapts its head for CIFAR-10.
    
    1. Loads the 100-class pretrained weights for all convolutional & LayerNorm stages.
    2. Swaps the final linear layer to `num_classes` (10 classes).
    3. Re-initializes the final layer deterministically.
    """
    effective_width = 8 if quick_test else width_base
    if quick_test:
        checkpoint_path = checkpoint_path.replace(".pt", "_quick.pt")

    if not os.path.exists(checkpoint_path):
        print(f"[Pretrain] Pretrained checkpoint not found at {checkpoint_path}. Training now...", flush=True)
        pretrain_resnet20_cifar100(
            output_path=checkpoint_path,
            epochs=1 if quick_test else 15,
            width_base=effective_width,
            device=device,
            seed=seed,
            quick_test=quick_test
        )


    # 1. Instantiate 100-class model to load weights
    base_model = ResNet20LN(num_classes=100, width_base=effective_width)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict)

    # 2. Instantiate 10-class model and transfer convolutional backbone weights
    model_cifar10 = ResNet20LN(num_classes=num_classes, width_base=effective_width)
    target_dict = model_cifar10.state_dict()

    # Copy all layers except the final classification layer ('fc.weight', 'fc.bias')
    for k, v in state_dict.items():
        if not k.startswith("fc."):
            target_dict[k] = v.clone()

    # Deterministically initialize the 10-class classification head
    torch.manual_seed(seed)
    in_features = model_cifar10.fc.in_features
    model_cifar10.fc = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model_cifar10.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model_cifar10.fc.bias, 0.0)

    model_cifar10.to(device)
    return model_cifar10
