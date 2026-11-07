"""Nested validation per held-out participant, over all LOSO folds.

Each outer fold retains one test subject and one validation subject, fitting all
normalization and weights on the remaining subjects. Computationally expensive
for real-world studies; no synthetic numbers should be published as evidence.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np

from crossguard_fi.config import TrainConfig
from crossguard_fi.data.manifest import load_manifest
from crossguard_fi.data.split import SubjectSplit, assert_disjoint
from crossguard_fi.training.engine import train, evaluate


def run_loso(manifest: Path, root: Path, output: Path, cfg: TrainConfig,
             max_folds: int | None = None) -> dict:
    records = load_manifest(manifest)
    subjects = sorted({rec.subject_id for rec in records})
    if len(subjects) < 3:
        raise ValueError("LOSO with independent validation needs >=3 subjects")
    if max_folds is not None and not 1 <= max_folds <= len(subjects):
        raise ValueError("max_folds must be between 1 and number of subjects")
    output.mkdir(parents=True, exist_ok=True)
    all_folds = []
    for idx, test_subject in enumerate(subjects[:max_folds], start=1):
        candidates = [s for s in subjects if s != test_subject]
        val_subject = random.Random(cfg.seed + idx).choice(candidates)
        outer_train = set(candidates) - {val_subject}
        split = SubjectSplit(
            train=tuple(r for r in records if r.subject_id in outer_train),
            val=tuple(r for r in records if r.subject_id == val_subject),
            test=tuple(r for r in records if r.subject_id == test_subject),
        )
        assert_disjoint(split)
        fold_dir = output / f"fold_{idx:03d}"
        train_report = train(manifest, root, fold_dir, cfg, split_override=split)
        result = evaluate(fold_dir / "best_model.pt", manifest, root,
                          fold_dir / "sealed_test.json")
        all_folds.append({"test_subject": test_subject, "val_subject": val_subject,
                          "test_macro_f1": result["macro_f1"],
                          "test_accuracy": result["accuracy"],
                          "best_epoch": train_report["best_epoch"]})
    scores = np.asarray([f["test_macro_f1"] for f in all_folds])
    summary = {"folds_completed": len(all_folds), "total_subjects": len(subjects),
               "complete_loso": len(all_folds) == len(subjects),
               "mean_subject_macro_f1": float(scores.mean()),
               "std_subject_macro_f1": float(scores.std(ddof=1)) if len(scores) > 1 else None,
               "folds": all_folds,
               "warning": "Synthetic results are only pipeline verification; external-domain tests still needed"}
    (output / "loso_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
