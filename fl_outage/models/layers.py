import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """2D Layer Normalization for image tensors [B, C, H, W].
    
    Normalizes across channels per spatial location and sample:
        y = (x - mean) / (std + eps) * weight + bias
    
    Crucial for Linear Mode Connectivity (LMC) and Model Merging:
    Unlike BatchNorm, LayerNorm does not track running batch statistics (mean/var).
    When two models are linearly interpolated (0.5 W_A + 0.5 W_B), the weights
    remain valid without needing activation statistics recalibration (e.g. REPAIR).
    """

    def __init__(self, num_channels: int, eps: float = 1e-7):
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)
        self.eps = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        # Normalize across channel dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply affine transformation
        out = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

    def __repr__(self) -> str:
        return f"LayerNorm2d({self.num_channels}, eps={self.eps})"
