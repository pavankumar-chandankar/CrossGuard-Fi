"""Defensive CSI ingestion: dimensional, numerical, temporal, and provenance checks."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from crossguard_fi.config import CONTRACT, CSIContract
from crossguard_fi.data.manifest import Recording, resolve_recording


class DataValidationError(ValueError):
    """Raised when an input recording fails the declared CSI contract."""


@dataclass(frozen=True)
class RecordingStats:
    samples: int
    channels: int
    duration_seconds: float
    max_gap_ns: int


def validate_csi(csi: np.ndarray, timestamps_ns: np.ndarray,
                 contract: CSIContract = CONTRACT) -> RecordingStats:
    if not isinstance(csi, np.ndarray) or csi.ndim != 2:
        raise DataValidationError("csi must have shape [time, channels]")
    if csi.shape[1] != contract.channels:
        raise DataValidationError(f"Expected {contract.channels} channels; got {csi.shape[1]}")
    if csi.shape[0] < contract.window_samples:
        raise DataValidationError("Recording is shorter than one analysis window")
    if csi.dtype.kind not in "fiu" or not np.all(np.isfinite(csi)):
        raise DataValidationError("CSI must contain finite numeric samples")
    if float(np.max(np.abs(csi))) > contract.max_abs_amplitude:
        raise DataValidationError("Amplitude exceeds defensive bound")
    if not isinstance(timestamps_ns, np.ndarray) or timestamps_ns.ndim != 1:
        raise DataValidationError("timestamps_ns must be a 1D integer array")
    if timestamps_ns.shape[0] != csi.shape[0] or timestamps_ns.dtype.kind not in "iu":
        raise DataValidationError("timestamp count/dtype mismatch")
    gaps = np.diff(timestamps_ns.astype(np.int64))
    expected = 1e9 / contract.sampling_hz
    if np.any(gaps <= 0):
        raise DataValidationError("Timestamps must be strictly increasing")
    if not np.all(np.abs(gaps - expected) <= expected * contract.sample_period_tolerance):
        raise DataValidationError("Sampling jitter/gap exceeds tolerance; resample before ingestion")
    return RecordingStats(
        samples=int(csi.shape[0]), channels=int(csi.shape[1]),
        duration_seconds=float((timestamps_ns[-1] - timestamps_ns[0]) * 1e-9),
        max_gap_ns=int(gaps.max()),
    )


def load_recording(root: Path, record: Recording,
                   contract: CSIContract = CONTRACT) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_recording(root, record)
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"csi", "timestamps_ns"}:
                raise DataValidationError("Expected npz arrays: csi and timestamps_ns")
            csi = data["csi"]
            ts = data["timestamps_ns"]
    except (OSError, KeyError, ValueError) as exc:
        raise DataValidationError(f"Could not safely load {path.name}: {exc}") from exc
    validate_csi(csi, ts, contract)
    return csi.astype(np.float32, copy=False), ts


def validate_dataset(root: Path, records: list[Recording]) -> dict:
    stats = [validate_csi(*load_recording(root, r)) for r in records]
    return {
        "recordings": len(records),
        "subjects": len({r.subject_id for r in records}),
        "domains": sorted({r.domain_id for r in records}),
        "classes_present": sorted({r.label for r in records}),
        "total_samples": sum(s.samples for s in stats),
        "min_duration_seconds": min(s.duration_seconds for s in stats),
        "max_gap_ns": max(s.max_gap_ns for s in stats),
    }
