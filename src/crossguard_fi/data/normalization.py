"""Train-only channel standardization; no subject/test statistics enter fitting."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.manifest import Recording
from crossguard_fi.data.validation import load_recording


@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray
    clip: float = 8.0

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        if (self.mean.shape != (CONTRACT.channels,)
                or self.std.shape != (CONTRACT.channels,)
                or not np.all(np.isfinite(self.mean))
                or not np.all(np.isfinite(self.std))
                or np.any(self.std <= 0)):
            raise ValueError("Invalid normalizer parameters")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, value: dict) -> "ChannelNormalizer":
        return cls(np.asarray(value["mean"]), np.asarray(value["std"]), float(value["clip"]))


def fit_normalizer(root: Path, records: tuple[Recording, ...]) -> ChannelNormalizer:
    if not records:
        raise ValueError("Cannot fit normalizer on empty training records")
    # Weighted across all training samples, independently of window overlap.
    count = 0
    total = np.zeros(CONTRACT.channels, dtype=np.float64)
    total_sq = np.zeros(CONTRACT.channels, dtype=np.float64)
    for record in records:
        csi, _ = load_recording(root, record)
        values = csi.astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_sq / count - mean ** 2, 1e-8)
    return ChannelNormalizer(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))
