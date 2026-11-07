# Pre-registered evaluation protocol for a future real-data study

## Research question

Can supervised 24-class recognition from 270-channel CSI windows generalize to previously unseen human subjects, and what degrades under room/radio/session shifts? A numerical improvement alone does not prove the network has learned body-motion representations.

## Primary split and metric

- **Primary partition:** subject-disjoint train/validation/sealed-test split from `subject_holdout` with deterministic random seed 17. Subject identifiers are grouped before any 2-second windows are formed. For modest subject counts, the default aims for about 60/20/20 (subject-rounded).
- **Model selection:** validation macro F1 over a fixed 24-class universe. No test peeking or selection by test F1.
- **Primary final metric:** sealed-test macro F1 with all 24 classes counted. Explicit support counts and confusion matrix accompany accuracy, present-class balanced accuracy and per-class precision/recall/F1.
- **Calibration:** NLL and ten-bin ECE, treated as descriptive metrics rather than a guarantee of safe confidence.
- **Seed control:** Python/NumPy/torch seeds, deterministic cuDNN settings, saved configuration and checkpoint metadata.

### Design limitations

The default subject holdout is a **single split**, not a statistically complete cross-subject study. The subject count and activity coverage of the real dataset may make this split inadequate. A publication-quality study should report repeated subject folds or full LOSO (implemented via `crossguard loso`), confidence intervals over subjects, external-room/device tests and repeated seeds. Do not generalize beyond the collection domains. If a rare activity has zero support in the test cohort, its per-class F1 is 0 by construction and that shortfall must be disclosed.

## Baselines and comparability

| Family | Evidence | Reason |
|---|---|---|
| Statistical mean/std MLP | implemented | interpretable small feature baseline |
| Residual dilated temporal CNN | implemented | local motion temporal patterns over 2 s |
| Majority class / classical RF/SVM | planned | control against spurious activity imbalance |
| CNN-LSTM / transformer | planned | temporal architecture comparison |
| Cross-room / cross-device | planned | environmental confounding and transfer |
| Signal ablations | planned | channel blocks, normalization, temporal filtering |

Compare identical splits and channel order, no augmentation in val/test, matching training budget and a documented hyperparameter search budget. Report parameter count, runtime, model size and latency on fixed hardware when real models exist. Multi-antenna/channel permutations should never be applied silently.

## Leakage prevention checklist

- [x] Subject partition before window extraction.
- [x] No train/validation/test subject overlap.
- [x] Normalization fit on train recordings only.
- [x] Checkpoint selects by validation, not test.
- [x] Test IDs stored in the checkpoint.
- [x] Test evaluation requires unchanged manifest digest.
- [x] Recording-level SHA-256 fingerprints captured at train time and validated before test evaluation.
- [ ] Independent room/radio evaluation cohorts (require source data).
- [ ] Verified class ontology and participant consent (require source data).
- [ ] Pre-registration of final hyperparameters before sealed test use (research procedure).
- [ ] Repeated random seeds and cross-subject statistical intervals (research procedure).

## Ablation and negative controls

Strong validation macro F1 can still be driven by room/subject confounding. Future ablations should include shuffled training labels (expected chance), sensor/channel-block shuffles, ID-only nuisance classifiers, subject/device prediction as a shortcut diagnostic, feature normalization alternatives, signal drift and device changes. All must be separately tracked, not implied by current code.

## Reporting standard

Every result table should include dataset version, sample frequency and window policy, subject counts, sample counts, hardware, model family, parameter count, seed, hyperparameters, class ontology, metric definition, checkpoint hash and confidence intervals where applicable. Never report synthetic-fixture accuracy as actual HAR accuracy.
