# Engineering decisions / ADRs

## ADR-001 — Declare geometry instead of guessing a public dataset

**Decision:** 270 CSI channels, 50 Hz and 24 classes are a project-specific target contract. The repository does not infer a subcarrier/antenna assignment or claim to have raw CSI samples. **Reason:** mapping guesses would create misleading RF semantics and unreproducible adapters.

## ADR-002 — Separate activity segments from window creation

**Decision:** one recording is a constant-label segment and windows never cross a recording boundary. **Reason:** avoids mixed-label windows, temporal boundary contamination and accidental leakage.

## ADR-003 — Group holdout by subject

**Decision:** split subjects before fitting anything; store test identities in the artifact. **Reason:** random window splits can reproduce near-duplicate movements, shared subject styles and radio environment artifacts in train/test. Subject holdout does not eliminate room/device confounding; external-domain evaluation remains necessary.

## ADR-004 — Train-only channel standardization

**Decision:** fit channel mean/std once on raw training recordings; freeze in checkpoint. **Reason:** overlapping windows can overweight some samples and fitting on test signals contaminates evaluation.

## ADR-005 — Two complementary model baselines

**Decision:** a residual temporal CNN and an MLP over channel statistics. **Reason:** compare temporal features against a simple statistical baseline; do not assume larger networks necessarily help.

## ADR-006 — Preserve native numerical evidence

**Decision:** C++17 Welford statistics, strict numeric parsing and a NumPy parity test at 270 channels. The model may use its own PyTorch representation. **Reason:** not conflating a fast native preprocessor with a neural inference backbone.

## ADR-007 — Fail readiness when a valid checkpoint is missing

**Decision:** no randomly initialized model serving fallback. **Reason:** 200 on liveness does not imply safe predictions. The ALB health path is `/readyz`, not `/healthz`.

## ADR-008 — Cloud deployment opt-in

**Decision:** Terraform and container release instructions are supplied, but no AWS resources or model data are provisioned. **Reason:** infrastructure is account-specific, billable, and must not be represented as already operating.
