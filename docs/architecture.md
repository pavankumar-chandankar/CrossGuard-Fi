# CrossGuard-Fi — system architecture and component boundaries

## Design principles

1. **Evidence before claims.** A workflow is implemented, tested, and identified as experimental; results require real, independently measured data.
2. **Separation of acquisition from supervised learning.** Acquiring/calibrating real CSI requires hardware, drivers and a dataset agreement; this repository begins at the versioned recording contract and does not pretend to control a Wi-Fi NIC.
3. **Record-level and subject-level isolation.** Every recording has an activity label, session and subject. Windows never cross recordings. Train/validation/test subjects do not overlap.
4. **Training/serving parity.** The same frozen train-only channel mean/std/clip settings are stored in the checkpoint and applied by the HTTP predictor.
5. **Model provenance.** Each artifact includes immutable data-schema expectations, an experiment seed, normalization settings, held-out identities, dataset-manifest digest, per-recording file SHA-256 fingerprints and model architecture metadata. The served artifact gets a SHA-256 identifier.
6. **Failure explicitness.** No silent NaN interpolation, no classification when the model is missing, no fallback to invented class labels, and no automatic network deployment.

## Component decomposition

### Recording ingestion

- `data/manifest.py` validates expected keys, IDs, version and safe relative paths; every file is resolved within the dataset root.
- `data/validation.py` rejects nonfinite data, wrong channel count, timestamp gaps, backwards time, invalid dtypes and recordings too short for a window.
- Manifest and data are **external artifacts**, not packaged alongside source. Real data should be handled under consent/licensing and retention policies.

### Experiment preparation

- `data/split.py` shuffles subject IDs deterministically and selects disjoint subject sets. It never splits individual windows randomly.
- `data/normalization.py` fits a mean and variance across **training recordings only**, once per raw sample (not overlapping windows).
- `data/dataset.py` builds a window index and reads underlying records on demand. A bounded per-worker cache avoids keeping the whole dataset resident, but NPZ files do not provide efficient range streaming. For terabyte-scale CSI, move to chunked Zarr/HDF5 shards after protocol review.

### PyTorch models

- `TemporalResNet`: 270-to-64 (configurable) projection; four residual dilated convolution blocks; mean/max temporal pooling and a 24-output classifier.
- `StatisticalMLP`: channel-wise mean/std over 100 samples (540 features), then a two-layer MLP; useful as a simpler baseline.
- Both models output logits, are trained with cross-entropy, can operate on CPU and support ordinary PyTorch gradient flows.

### Validation/test protocol

- The model checkpoint is saved on **validation macro F1** improvement; test-set membership is sealed in checkpoint metadata.
- Training reports omit test-set metrics. `crossguard evaluate` reopens the sealed partition only after training, checks the manifest digest and recording file hashes, and reports fixed-class macro F1, accuracy, present-class balanced accuracy, per-class scores, NLL, 10-bin ECE and a 24-by-24 confusion matrix.
- Repeated tuning against sealed-test output would invalidate the protocol. External replication should use fresh subjects, sessions, rooms and devices.

### Native C++ component

- The C++17 stream parser computes numerically stable Welford per-channel statistics. It has compile-time and runtime tests and produces no dependency on the Python training model.
- The C++ CLI is a feature-engineering baseline, not a CSI capture implementation. The MLP learns on matching mean/std statistics computed in torch; the `tests/test_cpp.py` checks C++ vs. NumPy numerical parity.

### Serving

- `Predictor` rejects model-contract mismatch and invalid inputs. Model weights are loaded via `torch.load(weights_only=True)`. No arbitrary checkpoint execution is needed.
- FastAPI exposes liveness and model readiness separately. Authenticated endpoints return probability-ranked classes and an artifact digest. Input is bound to 100 samples at 270 channels and a nominal 50 Hz timestamp sequence.
- The serving path intentionally doesn't return raw CSI in its response and doesn't log raw request bodies. The middleware checks declared body size; a reverse proxy must impose an independent actual-body limit and rate control.

### Operational deployment

- Docker compiles the native component and packages Python with CPU-only PyTorch. The final runtime is non-root, and Compose runs it read-only with a bind restricted to loopback.
- Terraform describes an HTTPS ALB, ECS Fargate tasks in private subnets, a scoped API-key secret and CloudWatch logging. AWS resources are **not provisioned** by a checkout or test command.

## Data-flow failure cases

| Failure | Behavior |
|---|---|
| Wrong number of channels | Reject recording/request with diagnostic |
| 50 Hz violated / missing samples | Reject input; require explicit documented resampling upstream |
| NaN/Inf in amplitude | Reject before normalization or prediction |
| Test subject appears in training | Raise leakage error |
| Dataset manifest changed after fitting | Abort test evaluation |
| Checkpoint architecture mismatch | Refuse model readiness |
| Wrong/missing API key | HTTP 401 on protected endpoints |
| Model absent | `/healthz` works; `/readyz` and inference return 503 |
| Model weights absent from release image | Container health check fails; never serve random weights |

## Non-goals of this version

Live CSI capture, antenna calibration, complex CSI phase processing, a validated 24-activity taxonomy, ML inference at an edge gateway, high-throughput streaming, tracking multi-person activity, benchmark claims, drift remediation and ongoing managed cloud deployment. They are meaningful research/engineering milestones, not hidden features.
