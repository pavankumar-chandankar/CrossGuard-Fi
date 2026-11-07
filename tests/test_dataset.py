from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest

from crossguard_fi.config import CONTRACT
from crossguard_fi.data.dataset import CSIDataset
from crossguard_fi.data.manifest import Recording, load_manifest, manifest_digest, resolve_recording
from crossguard_fi.data.normalization import fit_normalizer
from crossguard_fi.data.split import assert_disjoint, subject_holdout
from crossguard_fi.data.validation import validate_dataset


def test_manifest_and_disjoint_partitions(small_dataset):
    root, manifest = small_dataset
    records = load_manifest(manifest)
    assert len(records) == 72
    split = subject_holdout(records, seed=5)
    assert_disjoint(split)
    assert all(set(v).isdisjoint(set(u)) for k, v in split.subjects().items()
               for j, u in split.subjects().items() if k != j)
    assert subject_holdout(records, seed=5).subjects() == split.subjects()
    assert len(manifest_digest(records)) == 64


def test_validate_all_and_lazy_windows(small_dataset):
    root, manifest = small_dataset
    records = load_manifest(manifest)
    output = validate_dataset(root, records)
    assert output["recordings"] == 72
    assert output["classes_present"] == list(range(24))
    normalizer = fit_normalizer(root, tuple(records[:24]))
    dataset = CSIDataset(root, tuple(records[:24]), normalizer)
    assert len(dataset) == 24
    window, label = dataset[0]
    assert tuple(window.shape) == (100, 270)
    assert label.item() == 0
    assert np.all(np.isfinite(window.numpy()))


def test_train_only_normalization(small_dataset):
    root, manifest = small_dataset
    records = load_manifest(manifest)
    split = subject_holdout(records)
    norm = fit_normalizer(root, split.train)
    manual = np.concatenate([np.load(root / rec.relative_path)["csi"] for rec in split.train])
    assert np.allclose(norm.mean, manual.mean(axis=0), atol=1e-6)
    assert set(split.subjects()["train"]).isdisjoint(split.subjects()["test"])


def test_path_traversal_and_duplicate_records(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        Recording.from_mapping({"recording_id": "r", "relative_path": "../secret.npz",
                                "subject_id": "s", "session_id": "a", "domain_id": "b",
                                "label": 0})
    (tmp_path / "manifest.jsonl").write_text("\n".join([
        json.dumps({"recording_id": "r", "relative_path": "data/a.npz",
                    "subject_id": "s", "session_id": "a", "domain_id": "b", "label": 0})
    ] * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(tmp_path / "manifest.jsonl")


def test_subject_split_rejects_too_few_subjects(small_dataset):
    root, manifest = small_dataset
    records = [r for r in load_manifest(manifest) if r.subject_id == "s00"]
    with pytest.raises(ValueError):
        subject_holdout(records)
