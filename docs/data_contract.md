# Data contract, source provenance and adaptation protocol

## Declared target: 270 × 50 Hz × 24 classes

The project uses *user-stated* target dimensions. It does not assume 270 means a particular antenna/subcarrier layout; the true order, units, link mapping, amplitude transform, collection hardware and ground-truth taxonomy must come from the actual dataset. These characteristics affect physical interpretability and transfer performance.

| Field | Contract | Validation |
|---|---|---|
| `csi` | numeric array `[T, 270]` | finite, bounded, constant channel order |
| `timestamps_ns` | int64 length `T` | monotonically increasing, target step 20 ms ±15% |
| `label` | integer `0..23` for whole recording | no missing/unknown placeholder in supervised training |
| `recording_id` | globally unique string | duplicates rejected |
| `subject_id` | stable participant/pseudonym | train/test group isolation |
| `session_id` | stable collection session | helps audit same-subject environmental leakage |
| `domain_id` | room/device/environment identifier | supports planned external-domain evaluation |
| `relative_path` | `.npz` path under data root | absolute/path traversal/symlink escape rejected |
| `sampling_hz` | integer 50 | exact rate declaration |
| `schema_version` | integer 1 | exact match |

### Example manifest line

```json
{"recording_id":"participant_017_session_2_walk_003","relative_path":"recordings/participant_017_session_2_walk_003.npz","subject_id":"participant_017","session_id":"session_2","domain_id":"room_A-device_01","label":7,"sampling_hz":50,"schema_version":1}
```

`label=7` is an *arbitrary placeholder ID*; no activity name is inferred from that number.

### Recording writing example

```python
import numpy as np
np.savez_compressed(
    "recordings/example.npz",
    csi=csi.astype(np.float32),        # [samples, 270]
    timestamps_ns=timestamps.astype(np.int64),  # [samples], nominal 20,000,000 ns apart
)
```

The ingester uses `np.load(..., allow_pickle=False)` and does not execute serialized objects.

## Critical data preparation decisions before using real datasets

1. Document the antenna, receiving/transmitting radio, subcarrier and link ordering that produces 270 channels; preserve it across training and serving.
2. Choose and freeze a physical quantity (e.g., CSI amplitude) and units; log hardware firmware and data-collection software versions.
3. Associate each activity segment with verified ground truth and a stable class ontology; include a null/no-activity class only if it is explicitly one of the 24 verified labels.
4. Where gaps, resampling or clock drift occur, *segment or correct upstream* with a reproducible method and log original timestamps; avoid duplicating samples without disclosure.
5. Define a policy for invalid windows and zero-variance channels. This version rejects invalid samples and floors training variance to avoid division by zero.
6. Pseudonymize subject IDs. Retain an independent audit trail of original dataset filenames/checksums and license terms outside the repository.
7. Respect rights to use the source data; do not commit raw recordings, participant identifiers or private keys.

## Preprocessing parity

Per-channel mean/std is estimated once over all *training-recording* samples. Standardization is `(x - train_mean) / max(train_std, 1e-4)` clipped to `[-8,8]`. Held-out subjects never contribute to fitting. The resulting 270-channel vectors and clipping value are embedded in the model checkpoint and re-used at serving time.

## Window policy

A fixed 100-sample (2s) window and stride 50 (1s) are used. Only windows fully contained in a single recording are constructed; no window can combine labels, sessions or recordings. The current NPZ loader validates recordings fully on dataset construction; this imposes a cost for large datasets. No data augmentation is enabled by default, so any future augmentation experiment must be declared, ablated and restricted to training data.

## Synthetic fixture use

`crossguard generate-fixture` makes class-coded sinusoidal/noisy numerical arrays that intentionally have learnable patterns. This is only a *contract and training smoke test*. Any F1 score or latency from these fixtures is not a real-world Wi-Fi CSI result.
