"""Two fully reproducible, research-comparable PyTorch HAR baselines."""
import torch
from torch import nn

from crossguard_fi.config import CONTRACT


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation,
                      dilation=dilation),
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                      dilation=dilation),
            nn.GroupNorm(min(8, width), width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.network(x))


class TemporalResNet(nn.Module):
    """Dilated residual 1D CNN with temporal mean/max pooling; input [B, T, 270]."""
    def __init__(self, width: int = 64, dropout: float = 0.15):
        super().__init__()
        if width % 8 != 0:
            raise ValueError("TCN width must be divisible by 8")
        self.project = nn.Sequential(nn.Conv1d(CONTRACT.channels, width, 1), nn.GELU())
        self.backbone = nn.Sequential(*[
            ResidualTemporalBlock(width, dilation=d, dropout=dropout)
            for d in (1, 2, 4, 8)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Dropout(dropout),
            nn.Linear(width * 2, CONTRACT.classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (CONTRACT.window_samples, CONTRACT.channels):
            raise ValueError("Expected [batch, 100, 270]")
        z = self.backbone(self.project(x.transpose(1, 2)))
        z = torch.cat((z.mean(dim=-1), z.amax(dim=-1)), dim=1)
        return self.head(z)


class StatisticalMLP(nn.Module):
    """Feature-engineering baseline: channel-wise mean/std, then 2-layer MLP."""
    def __init__(self, width: int = 64, dropout: float = 0.15):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(CONTRACT.channels * 2, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, CONTRACT.classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (CONTRACT.window_samples, CONTRACT.channels):
            raise ValueError("Expected [batch, 100, 270]")
        features = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=1)
        return self.classifier(features)


def build_model(name: str, hidden_dim: int = 64, dropout: float = 0.15) -> nn.Module:
    if name == "tcn":
        return TemporalResNet(hidden_dim, dropout)
    if name == "mlp":
        return StatisticalMLP(hidden_dim, dropout)
    raise ValueError(f"Unknown model: {name}")
