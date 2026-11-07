# CrossGuard-Fi

**A reproducible engineering reference for Wi-Fi Channel State Information (CSI) human-activity classification: 270 input channels, 50 Hz sampling, 24 output classes.**

[![CrossGuard-Fi research-engineering CI](https://github.com/pavankumar-chandankar/CrossGuard-Fi/actions/workflows/ci.yml/badge.svg)](https://github.com/pavankumar-chandankar/CrossGuard-Fi/actions/workflows/ci.yml)

> **Status: in development.** This repository implements a coherent software system—from validated recording manifests and subject-disjoint training to PyTorch inference, C++17 feature extraction, a FastAPI interface, containers and AWS ECS/Terraform templates. It does **not** contain real Wi-Fi CSI measurements, a fitted production classifier, a published 24-class benchmark, or a live AWS deployment. Synthetic fixtures and their results are strictly functional tests, never evidence of real recognition performance.

## 1. Why this architecture?

Wi-Fi CSI pipelines can appear to achieve excellent classification by memorizing room/device/subject artifacts rather than recognizing an activity. A deployable system also needs input contracts, training/serving parity, versioned models, failure-safe inference, and reproducible evaluation. CrossGuard-Fi separates these concerns and makes them testable rather than describing a single notebook as a finished ML product.

### Requirements and implementation traceability

| Requirement | Implemented location | Verification |
|---|---|---|
| 270 CSI channels / 50 Hz / 24 activity IDs | `config.py`, `data/validation.py` | dimensional, timestamp, label tests |
| Per-recording data validation | `data/manifest.py`, `validation.py` | malformed, traversal, duplicate, NaN, gap tests |
| 2s sliding windows, 50% overlap | `data/dataset.py` | window/subject partition tests |
| Subject-disjoint evaluation | `data/split.py`, `training/engine.py` | deterministic splits, no overlap assertions |
| Training-only normalization | `data/normalization.py` | independent mean comparison |
| PyTorch neural and feature baselines | `models/networks.py` | forward/backward and shape tests |
| Reproducible train/checkpoint/test | `training/` | end-to-end smoke with digest verification |
| Accuracy, macro F1, per-class, calibration | `training/metrics.py` | confusion/calibration tests |
| C++17 deterministic CSI statistics | `cpp/csi_features.cpp` | g++, Clang, 270-channel NumPy parity |
| Local/HTTP inference | `serving/predictor.py`, `serving/api.py` | API auth, validation, readiness tests |
| Container packaging | `Dockerfile`, `compose.yaml` | local Docker build/run instructions |
| AWS deployment design | `infra/terraform/` | HTTPS ALB + private Fargate + Secrets Manager IaC |
| Security, operations, protocols | `docs/` | requirements and known limitations documented |

## 2. Architecture

```text
  Recorded CSI files (.npz) + versioned manifest.jsonl
          | [schema / 270 channels / 50Hz timing / finite values]
          v
    Subject-disjoint splits (train / validation / sealed test)
          |                    (no window crosses a recording)
          v
    Train-only channel statistics -> 2-second windows -> PyTorch
          |                                      +----------------+
          |                                      | TCN (residual) |
          |                                      | statistical MLP|
          |                                      +----------------+
          v
    Validation-selected checkpoint (metadata, split IDs, manifest and file SHA-256 digests)
          |                     |
          v                     +--> sealed subject test (reported separately)
    Strict Predictor -> FastAPI -> Docker -> optional AWS ECS/Fargate + ALB
          ^
    C++17 CSI feature tool (independent deterministic numeric baseline)
```

The C++ program is a tested feature-extraction component; the PyTorch TCN operates on time-series windows directly. **The C++ statistics are not falsely described as feeding the TCN in production.** The parallel statistical MLP has the matching mean/std feature structure but computes it in PyTorch; `tests/test_cpp.py` checks numerical parity between native extraction and NumPy.

## 3. Data contract — fixed targets, not an invented dataset

- **Input stream:** 270 finite numeric CSI amplitude channels per time step, at **50 Hz** (nominal 20 ms timestamps).
- **Window:** 100 chronological samples = 2 seconds; **stride 50** samples; 24 integer labels `[0, 23]`.
- **Recording:** one constant-label activity segment per `.npz` file, containing arrays `csi: float32[T,270]` and `timestamps_ns: int64[T]`.
- **Manifest:** JSON Lines with `recording_id`, relative file path, `subject_id`, `session_id`, `domain_id`, `label`, and schema metadata.
- **Recording constraints:** amplitude bound, finite values, monotonic timestamps and ±15% inter-sample timing tolerance. Records with large gaps must be resampled/segmented **upstream with documented provenance**, not silently repaired.
- **Ontology:** `activity_00`...`activity_23` are explicit placeholders; no real activity names are invented. A real dataset must define validated labels, antenna/subcarrier ordering, calibration, and ground-truth procedures.

See [data contract](docs/data_contract.md) for schema and missing-sample policy.

## 4. Installation (Python 3.11+, local CPU/GPU)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
pytest -q
```

The project uses PyTorch, FastAPI, NumPy and Pydantic. The C++17 extractor has **no external C++ runtime dependencies**. The repository supports local CPU operation; GPU training is optional. There is no cloud dependency for tests or training.

## 5. Demonstration: real workflow using clearly synthetic fixtures

**The following synthetic data are manufactured for software verification. Do not cite the training numbers as HAR performance.**

```bash
crossguard generate-fixture --output datasets/synthetic --subjects 4
crossguard validate --manifest datasets/synthetic/manifest.jsonl \
  --data-root datasets/synthetic
crossguard train --manifest datasets/synthetic/manifest.jsonl \
  --data-root datasets/synthetic --output runs/demo --model tcn \
  --epochs 3 --batch-size 16 --hidden-dim 32 --device cpu
crossguard evaluate --manifest datasets/synthetic/manifest.jsonl \
  --data-root datasets/synthetic --checkpoint runs/demo/best_model.pt \
  --output runs/demo/sealed_test.json
crossguard predict --checkpoint runs/demo/best_model.pt \
  --recording datasets/synthetic/recordings/s00_c00_t00.npz
```

For a full subject-wise outer test procedure with separate validation subjects, run:

```bash
crossguard loso --manifest datasets/synthetic/manifest.jsonl \
  --data-root datasets/synthetic --output runs/loso \
  --model mlp --epochs 2 --hidden-dim 32
```

This processes all subjects unless `--max-folds` is passed, and writes individual fold reports and a subject-macro-F1 summary. It is a software research protocol, not a claim of external domain generalization.

The first command generates only a few 2-second class-coded artificial recordings. Train/evaluation code is identical to the code used for real data once the approved real-data adapter is provided.

**Fast offline proof that all stages work:**

```bash
python scripts/smoke.py
```

That script creates temporary synthetic data, trains one epoch of the small MLP, checks a frozen test partition, tests model loading, and verifies inference. It automatically deletes temporary fixture data after completion.

## 6. Native C++17 feature extraction

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
printf '1,2\n3,4\n' | ./build/csi_features 2
```

The CLI takes a channel count and headerless CSV on stdin, and emits per-channel population mean/stddev. It rejects invalid counts, malformed rows, NaN/Inf and empty input. Test parity for 270 channels with:

```bash
pytest -q tests/test_cpp.py
clang++ --analyze -std=c++17 -Wall -Wextra cpp/csi_features.cpp
```

The original `python/pipeline.py` feature-CLI format is retained as a compatibility adapter; it is not the production training code.

## 7. Inference serving

The server refuses readiness unless an actual **contract-compatible** checkpoint is loaded. Local model loading uses PyTorch's safer `weights_only=True` mechanism. It does not execute arbitrary checkpoint pickles.

```bash
export CGF_MODEL_PATH="$PWD/runs/demo/best_model.pt"
export CGF_API_KEY="replace-with-a-random-secret"
export CGF_ENV=production
uvicorn crossguard_fi.serving.api:app --host 127.0.0.1 --port 8000
```

- `GET /healthz`: process liveness, independent of model availability.
- `GET /readyz`: HTTP 200 only if compatible model is loaded; otherwise 503.
- `GET /v1/model`: model digest, model family, ontology placeholders and contract (authenticated).
- `POST /v1/predict`: one `[100,270]` window and 100 `timestamps_ns` integers, plus `top_k` (authenticated).
- `POST /v1/predict-batch`: up to eight windows; validated and processed sequentially (authenticated).

To build an example request and call the service:

```bash
python scripts/build_api_request.py \
  --recording datasets/synthetic/recordings/s00_c00_t00.npz --output /tmp/cgf-request.json
curl --fail-with-body -H 'Content-Type: application/json' \
  -H "X-API-Key: $CGF_API_KEY" --data-binary @/tmp/cgf-request.json \
  http://127.0.0.1:8000/v1/predict
```

Input validation rejects wrong tensor shapes, nonfinite CSI values, excessive amplitudes, and timestamp gaps. Inference metadata includes the checkpoint SHA-256 for traceability. **API-key authentication is a minimal defense only**; consult the [threat model](docs/threat_model.md) for TLS, quotas, monitoring, secrets and privacy gaps.

## 8. Container workflow

A release checkpoint must already exist locally. Save it to `artifacts/model.pt` before building; **do not add it to Git**.

```bash
cp runs/demo/best_model.pt artifacts/model.pt
export CGF_API_KEY="replace-with-a-random-local-secret"
docker compose up --build
```

The container runs as a non-root user, has a read-only filesystem in Compose, a disabled-by-default internet-facing bind (`127.0.0.1` only), and health checking through `/readyz`. The Docker build includes CPU-only PyTorch and compiles the native extractor. Without a model checkpoint, readiness/healthchecks fail **by design**.

## 9. Optional AWS deployment (not deployed or charged)

`infra/terraform/` describes a deployment to **ECS Fargate in private subnets** with a public HTTPS Application Load Balancer, CloudWatch logging, Secrets Manager API-key injection, scoped security groups, and immutable ECR image digests. The deployment requires your own AWS account, existing VPC/subnets, ACM certificate, ECR image and secret. It was **not deployed, cost-tested or load-tested** here.

**AWS resources incur real charges** (Fargate, ALB, NAT/endpoints, logs, etc.). Nothing in this repository provisions resources by default or creates a free-tier promise. See [AWS deployment](docs/aws_deployment.md) for prerequisites, image publication, TLS, risk, cleanup and cost controls.

## 10. Evaluation integrity and production gaps

- **Implemented:** subject-disjoint split, immutable test membership stored in checkpoints, train-only normalization, deterministic seeding, class-complete macro F1, calibration ECE, confusion matrix, CPU inference and basic API control.
- **Not evaluated:** real dataset accuracy, per-room/per-device generalization, RF link variations, noisy packet acquisition, physical deployment throughput, drift, calibration reliability, interpretability, adversarial robustness or real AWS service costs.
- **Recommended next work:** obtain dataset/permission and channel ordering, verify label ontology, add site-disjoint and cross-device external test cohorts, capture operational latency and cold-start, assess class imbalance and domain shifts, add authentication gateway/rate limits, pin dependency lockfiles and sign release model artifacts.

Further reading: [system architecture](docs/architecture.md), [experiment protocol](docs/experiment_protocol.md), [model card](docs/model_card.md), [threat model](docs/threat_model.md), [operations](docs/operations.md), and [engineering decisions](docs/decisions.md).

## CV-safe description

> **CrossGuard-Fi — Wi-Fi CSI preprocessing and ML inference (in development):** Implemented a Python/PyTorch research pipeline targeting 270 CSI channels at 50 Hz and 24 activity classes, including validated recording manifests, subject-disjoint training/evaluation, reproducible checkpoints, C++17 feature extraction with numerical regression tests, FastAPI inference, Docker packaging, and AWS ECS/Terraform deployment configuration. Real-data accuracy and AWS deployment are pending.

This describes what is implemented and distinguishes **deployment configuration** from **production deployment**.

**Maintainer:** Pavankumar Chandankar. Repository code changes and demonstrations should be assessed by their source, tests and CI history; this project does not purport to be a published PhD system.
