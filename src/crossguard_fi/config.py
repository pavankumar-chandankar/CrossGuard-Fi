"""Immutable project-level data and experiment contracts.

The channel count and activity count are user-specified engineering targets, not
an assertion about an existing publicly available dataset or trained weights.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CSIContract:
    channels: int = 270
    sampling_hz: int = 50
    classes: int = 24
    window_samples: int = 100  # 2 seconds of 50 Hz signal
    stride_samples: int = 50  # 50% overlap, within a recording only
    max_abs_amplitude: float = 1e6
    sample_period_tolerance: float = 0.15
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (self.channels <= 0 or self.sampling_hz <= 0 or self.classes <= 1
                or self.window_samples < 4 or self.stride_samples <= 0):
            raise ValueError("Invalid CSI contract")
        if not 0 < self.sample_period_tolerance < 1:
            raise ValueError("Invalid sampling tolerance")

    def to_dict(self) -> dict:
        return asdict(self)


CONTRACT = CSIContract()


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    model: str = "tcn"
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 4
    hidden_dim: int = 64
    dropout: float = 0.15
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.model not in {"tcn", "mlp"}:
            raise ValueError("model must be tcn or mlp")
        if (self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0
                or self.patience < 1 or self.hidden_dim < 8 or not 0 <= self.dropout < 1):
            raise ValueError("Invalid training hyperparameters")
