"""Safe checkpoint loading and strictly bounded model-facing input contract."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.normalization import ChannelNormalizer
from crossguard_fi.training.checkpoints import load_checkpoint


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Serving device must be cpu or cuda")
        self.model, self.metadata = load_checkpoint(checkpoint, device=device)
        self.normalizer = ChannelNormalizer.from_dict(self.metadata["normalizer"])
        self.device = device
        self.model_sha256 = sha256(Path(checkpoint).read_bytes()).hexdigest()

    @staticmethod
    def check_input(samples: np.ndarray, timestamps_ns: np.ndarray):
        if samples.shape != (CONTRACT.window_samples, CONTRACT.channels):
            raise ValueError("Inference expects exactly 100 samples of 270 channels")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Non-finite CSI input")
        if np.max(np.abs(samples)) > CONTRACT.max_abs_amplitude:
            raise ValueError("CSI amplitude exceeds bound")
        if timestamps_ns.shape != (CONTRACT.window_samples,):
            raise ValueError("Expected 100 timestamps")
        gaps = np.diff(timestamps_ns.astype(np.int64))
        period = 1e9 / CONTRACT.sampling_hz
        if np.any(gaps <= 0) or np.any(abs(gaps - period) > period * CONTRACT.sample_period_tolerance):
            raise ValueError("Timestamps violate 50 Hz sampling contract")

    def predict(self, samples: list[list[float]] | np.ndarray,
                timestamps_ns: list[int] | np.ndarray, top_k: int = 3) -> dict:
        if not 1 <= top_k <= CONTRACT.classes:
            raise ValueError("top_k must be between 1 and 24")
        try:
            raw = np.asarray(samples, dtype=np.float32)
            timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Non-numeric input") from exc
        self.check_input(raw, timestamps)
        standardized = self.normalizer.transform(raw)
        tensor = torch.from_numpy(standardized[None]).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0].cpu().numpy()
        indices = np.argsort(-probabilities, kind="stable")[:top_k]
        return {
            "predicted_class": int(indices[0]),
            "confidence": float(probabilities[indices[0]]),
            "top_k": [
                {"class_id": int(i), "class_name": self.metadata["label_names"][int(i)],
                 "probability": float(probabilities[int(i)])} for i in indices
            ],
            "model_sha256": self.model_sha256,
            "contract_version": CONTRACT.schema_version,
        }
