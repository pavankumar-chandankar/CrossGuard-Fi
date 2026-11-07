"""Deterministic single-seed training, validation, and locked test evaluation."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from crossguard_fi.config import CONTRACT, TrainConfig
from crossguard_fi.data.dataset import CSIDataset
from crossguard_fi.data.manifest import load_manifest, manifest_digest, file_digests
from crossguard_fi.data.normalization import fit_normalizer, ChannelNormalizer
from crossguard_fi.data.split import subject_holdout, SubjectSplit, assert_disjoint
from crossguard_fi.models.networks import build_model
from crossguard_fi.training.checkpoints import load_checkpoint, save_checkpoint
from crossguard_fi.training.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def collect_probabilities(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_probabilities = [], []
    with torch.inference_mode():
        for inputs, labels in loader:
            prediction = model(inputs.to(device)).softmax(dim=1)
            all_probabilities.append(prediction.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probabilities)


def train(manifest: Path, dataset_root: Path, output_dir: Path,
          cfg: TrainConfig = TrainConfig(),
          split_override: SubjectSplit | None = None) -> dict:
    set_seed(cfg.seed)
    records = load_manifest(manifest)
    split = split_override if split_override is not None else subject_holdout(records, seed=cfg.seed)
    assert_disjoint(split)
    if sorted(r.recording_id for r in (*split.train, *split.val, *split.test)) != sorted(
            r.recording_id for r in records):
        raise ValueError("Split must account for each manifest recording exactly once")
    absent = set(range(CONTRACT.classes)) - {r.label for r in split.train}
    if absent:
        raise ValueError(f"Training split is missing classes: {sorted(absent)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = fit_normalizer(dataset_root, split.train)
    train_data = CSIDataset(dataset_root, split.train, normalizer)
    val_data = CSIDataset(dataset_root, split.val, normalizer)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    device = choose_device(cfg.device)
    model = build_model(cfg.model, cfg.hidden_dim, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    best_f1, best_epoch, epochs_without_gain = -1.0, 0, 0
    history: list[dict] = []
    digest = manifest_digest(records)
    source_sha256 = file_digests(dataset_root, records)
    metadata = {
        "model": cfg.model, "hidden_dim": cfg.hidden_dim, "dropout": cfg.dropout,
        "seed": cfg.seed, "manifest_digest": digest,
        "recording_sha256": source_sha256,
        "normalizer": normalizer.to_dict(), "subject_split": split.subjects(),
        "recording_split": {part: [r.recording_id for r in getattr(split, part)]
                            for part in ("train", "val", "test")},
        "label_names": [f"activity_{i:02d}" for i in range(CONTRACT.classes)],
        "note": "Class names are placeholders; replace with verified dataset ontology before deployment",
    }
    checkpoint = output_dir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        started = time.perf_counter()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            seen += len(targets)
        labels, probabilities = collect_probabilities(model, val_loader, device)
        results = classification_metrics(labels, probabilities)
        record = {"epoch": epoch, "training_loss": running_loss / seen,
                  "validation_macro_f1": results["macro_f1"],
                  "validation_accuracy": results["accuracy"],
                  "elapsed_seconds": round(time.perf_counter() - started, 3)}
        history.append(record)
        if results["macro_f1"] > best_f1 + 1e-9:
            best_f1 = results["macro_f1"]
            best_epoch = epoch
            epochs_without_gain = 0
            save_checkpoint(checkpoint, model, metadata)
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= cfg.patience:
            break
    report = {
        "artifact_type": "validation_only", "dataset_manifest_sha256": digest,
        "training_parameters": asdict(cfg), "contract": CONTRACT.to_dict(),
        "train_subjects": split.subjects()["train"],
        "validation_subjects": split.subjects()["val"],
        "sealed_test_subjects": split.subjects()["test"],
        "windows": {"train": len(train_data), "val": len(val_data)},
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "history": history,
        "warning": "Synthetic-fixture performance is not empirical Wi-Fi HAR evidence. Test set untouched.",
    }
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(checkpoint: Path, manifest: Path, dataset_root: Path,
             output_path: Path, batch_size: int = 32, device: str = "cpu") -> dict:
    records = load_manifest(manifest)
    model, metadata = load_checkpoint(checkpoint, device=choose_device(device))
    if manifest_digest(records) != metadata["manifest_digest"]:
        raise ValueError("Manifest changed since training; abort sealed-test evaluation")
    if file_digests(dataset_root, records) != metadata.get("recording_sha256"):
        raise ValueError("Recording bytes changed since training; abort sealed-test evaluation")
    ids = set(metadata["recording_split"]["test"])
    heldout = tuple(r for r in records if r.recording_id in ids)
    if len(heldout) != len(ids):
        raise ValueError("Test partition no longer matches checkpoint")
    normalizer = ChannelNormalizer.from_dict(metadata["normalizer"])
    test_data = CSIDataset(dataset_root, heldout, normalizer)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    labels, probabilities = collect_probabilities(model, loader, choose_device(device))
    report = classification_metrics(labels, probabilities)
    report["artifact_type"] = "sealed_subject_test"
    report["test_subjects"] = sorted({r.subject_id for r in heldout})
    report["dataset_manifest_sha256"] = metadata["manifest_digest"]
    report["warning"] = "Do not tune hyperparameters against this test report."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
