# Model card (template — no real model released)

**Model status:** No production-trained weights included. The code supports TemporalResNet and StatisticalMLP examples. No independent real-data performance is asserted.

- **Intended use:** research and engineering evaluation of *single-label* activity classification from calibrated, appropriately collected, consented 270-channel CSI amplitude streams at 50 Hz.
- **Not intended for:** surveillance of people without consent, medical/safety-critical decisions, identification, detection of private activities, legal or disciplinary decisions, or making claims about personal identity.
- **Input:** 100 × 270 numeric amplitude window and monotonic timestamps at 50 Hz.
- **Output:** 24 logits transformed into probabilities, with class IDs. Class names are placeholders pending real dataset ontology.
- **Validation:** subject-disjoint splits, train-only standardization, calibration/NLL and per-class metrics implemented. Actual measured performance: **not available**.
- **Known uncertainty:** cross-room/device/domain transfer, channel order, measurement noise, class coverage, false positives, class imbalance, demographic or situational biases, and confidence calibration.
- **Privacy:** radio measurements may reveal human behavior or location-related patterns. Apply collection permission, retention limits and access control to all sensor traces.
- **Reproducibility:** seeded experiments, source code, manifests, digest of manifest, metadata and checkpoint hash. A signed release and dataset file integrity process remain to be added.
- **Release criteria:** confirm ontology, verify measurement provenance, complete external room/subject evaluations, test latency and operational capacity, run security review, audit licensing, obtain deployment consent.

## Release evidence table

| Release item | Current status |
|---|---|
| Source code + tests | Implemented |
| Synthetic end-to-end smoke | Implemented |
| Real CSI training dataset | Not supplied |
| Real-data metrics and confidence intervals | Not available |
| Production weight artifact | Not available |
| External-domain replication | Not available |
| AWS production environment | Not deployed |
| Model governance and monitoring | Partial design only |
