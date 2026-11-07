# Operations, runbook and verification gates

## Local build and dataset gates

From a clean checkout:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/smoke.py
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

All tests and smoke fixtures are synthetic, CPU-safe and require no real CSI recordings or AWS resources.

### Dataset intake

- Record the exact data source/license, collection environment, number of participants and device/radio/antenna/subcarrier map.
- Validate subject/session/domain IDs and non-overlapping class label segments; record label ontology version, annotator QA and missing-data policy.
- Generate `.npz` files under a private dataset directory and write a JSONL manifest; run `crossguard validate`.
- Do not commit raw data. Train-only normalization is part of the pipeline; do not normalize real data before subject splitting.

### Experiment lifecycle

1. Freeze a manifest revision and run `crossguard validate`.
2. Run `crossguard train` with a declared seed/model budget; archive `train_report.json`, checkpoint and source commit.
3. Choose the best model using **validation only**. If testing alternative architectures, compare within the same train/validation split and budget.
4. Once frozen, run `crossguard evaluate` on the sealed test subject cohort. The evaluator checks both manifest and recording bytes SHA-256 against the training artifact.
5. For subject-level error bars, run `crossguard loso` and report all individual outer folds; do not conflate LOSO testing with independent room/device generalization.
6. Populate a real model card with dataset cohorts, measured accuracy/macro F1, calibration, failure analysis, edge latency and caveats.

### Incident response

| Event | Response |
|---|---|
| API `/healthz` 200 but `/readyz` 503 | Verify checkpoint exists/loads, contract matches and artifact path is correct; do not bypass readiness |
| Input shape/timestamp validation 422 | Quarantine misaligned recording and investigate upstream hardware/channel ordering; do not impute without protocol |
| Unexpected class-distribution change | Stop treating confidence as trustworthy; inspect source devices/sites and compare against test cohorts |
| High latency / out-of-memory | Check input sizes, batch size, CPU worker count and image footprint; do not alter evaluation claims without rerun |
| Key compromise | Revoke/rotate secret, force new ECS deployment, review IAM and access logs |
| Suspected data leakage | Halt result publication, reproduce split and subject audit, rerun from clean partitions |
| Unexpected AWS spend | Scale/remove service, check ALB, NAT, logs, Secrets Manager, ECR and remaining state; verify account budget alarms |

### Monitoring requirements (not implemented as live alarms)

A credible future service needs p50/p95/p99 request latency, request rejection counts, 4xx/5xx rates, concurrent requests, response-class proportions, drift by environment/radio, model load failures, CPU/memory saturation, deployment age and error budgets. Avoid logging raw CSI/subject identifiers. Configure retention, budget alarms and incident escalation with the responsible operator.

### Release and rollback

Release artifacts should be tied to a Git commit, dataset manifest digest, per-recording content digests, configuration, PyTorch and compiler versions, a signed checkpoint, immutable container digest and a documented model-card approval. Roll back using a previously validated image digest plus preserved secret/configuration version. The supplied Terraform is a *design template*, not an automated production release or disaster-recovery test.
