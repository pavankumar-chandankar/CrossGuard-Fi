"""Deterministic synthetic CI fixture; not representative of real Wi-Fi CSI or model accuracy."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.manifest import Recording


def generate_fixture(output: Path, subjects: int = 4, trials_per_class: int = 1,
                     samples: int = 110, seed: int = 17,
                     classes: int = CONTRACT.classes) -> Path:
    if subjects < 3 or trials_per_class < 1 or samples < CONTRACT.window_samples:
        raise ValueError("Need >=3 subjects and >=100 samples per trial")
    if not 1 <= classes <= CONTRACT.classes:
        raise ValueError("classes must be in 1..24")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "recordings"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    period_ns = int(1_000_000_000 / CONTRACT.sampling_hz)
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for subject in range(subjects):
            for cls in range(classes):
                for trial in range(trials_per_class):
                    # Deliberately simple synthetic class signatures: ONLY for CI.
                    phase = np.linspace(0, 2 * np.pi, samples, dtype=np.float32)
                    csi = rng.normal(0, 0.16, (samples, CONTRACT.channels)).astype(np.float32)
                    carrier = (np.sin(phase[:, None] * (1 + cls % 5))
                               * np.linspace(0.03, 0.12, CONTRACT.channels)[None, :])
                    csi += carrier.astype(np.float32)
                    csi[:, cls * 10:(cls + 1) * 10] += 0.6
                    csi += (subject - subjects / 2) * 0.012
                    identifier = f"s{subject:02d}_c{cls:02d}_t{trial:02d}"
                    path = data_dir / f"{identifier}.npz"
                    timestamps = np.arange(samples, dtype=np.int64) * period_ns
                    np.savez_compressed(path, csi=csi, timestamps_ns=timestamps)
                    record = Recording(
                        recording_id=identifier, relative_path=f"recordings/{identifier}.npz",
                        subject_id=f"s{subject:02d}", session_id=f"trial_{trial:02d}",
                        domain_id="synthetic_lab", label=cls,
                    )
                    handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return manifest_path
