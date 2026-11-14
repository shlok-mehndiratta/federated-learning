from collections import OrderedDict
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_outage.models.layers import LayerNorm2d


class ResNetBlock(nn.Module):
    """Residual block with LayerNorm instead of BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        # using layernorm instead of batchnorm so client running stats dont diverge
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = LayerNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        # skip connection shortcut matching dimensions
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ('norm', LayerNorm2d(out_channels))
            ]))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # foward pass through convs and add residual skip
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act2(out)


class ResNet20LN(nn.Module):
    """ResNet-20 for CIFAR-10 with Layer Normalization.
    
    Structure:
    - Initial Conv (3 -> 16, 3x3) + LayerNorm + ReLU
    - Stage 1: 3 blocks (16 filters)
    - Stage 2: 3 blocks (32 filters, first block downsamples stride 2)
    - Stage 3: 3 blocks (64 filters, first block downsamples stride 2)
    - Global Average Pooling -> Linear (64 -> num_classes)
    
    Total paramaters for num_classes=10: 272,474.
    """

    def __init__(self, num_classes: int = 10, width_base: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.width_base = width_base

        # 3 stages, 3 blocks each = 18 conv layers + 1 init conv + 1 fc = ResNet-20
        plan = [(width_base * (2**i), 3) for i in range(3)]
        self.plan = plan

        curr_filters = plan[0][0]
        self.init_conv = nn.Conv2d(3, curr_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.init_norm = LayerNorm2d(curr_filters)
        self.init_act = nn.ReLU(inplace=True)

        blocks = []
        for stage_idx, (out_filters, num_blocks) in enumerate(plan):
            for block_idx in range(num_blocks):
                downsample = (stage_idx > 0 and block_idx == 0)
                blocks.append(ResNetBlock(curr_filters, out_filters, downsample=downsample))
                curr_filters = out_filters
        self.stages = nn.Sequential(*blocks)

        self.fc = nn.Linear(curr_filters, num_classes)
        # initialize weights before trainning
        self.reset_parameters()

    def reset_parameters(self):
        # standard kaiming initializtion for conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, LayerNorm2d):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass through initial conv and 3 residual stages
        out = self.init_act(self.init_norm(self.init_conv(x)))
        out = self.stages(out)
        # global average pooling before classification head
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.fc(out)


def get_model(model_name: str = "resnet20_ln", num_classes: int = 10) -> nn.Module:
    """Model factory."""
    if model_name.lower() in ("resnet20_ln", "resnet20", "resnet20-32"):
        return ResNet20LN(num_classes=num_classes, width_base=16)
    raise ValueError(f"Unknown model name: {model_name}")
