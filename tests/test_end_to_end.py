import json
from pathlib import Path
import numpy as np
import pytest

from crossguard_fi.config import CONTRACT, TrainConfig
from crossguard_fi.data.manifest import load_manifest
from crossguard_fi.serving.predictor import Predictor
from crossguard_fi.training.engine import evaluate, train


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory):
    from crossguard_fi.data.synthetic import generate_fixture
    root = tmp_path_factory.mktemp("crossguard-training")
    manifest = generate_fixture(root, subjects=3, samples=100, seed=19)
    output = root / "run"
    report = train(manifest, root, output, TrainConfig(
        model="mlp", epochs=1, batch_size=12, hidden_dim=16, seed=19, device="cpu"))
    return root, manifest, output, report


def test_training_evaluation_provenance(model_artifact):
    root, manifest, output, report = model_artifact
    assert report["best_epoch"] == 1
    assert len(report["sealed_test_subjects"]) == 1
    checkpoint = output / "best_model.pt"
    assert checkpoint.exists()
    test = evaluate(checkpoint, manifest, root, output / "test_report.json")
    assert test["artifact_type"] == "sealed_subject_test"
    assert test["samples"] == 24
    assert len(test["confusion_matrix"]) == 24
    assert report["sealed_test_subjects"] == test["test_subjects"]


def test_predictor_and_ontology(model_artifact):
    root, manifest, output, report = model_artifact
    predictor = Predictor(output / "best_model.pt")
    record = load_manifest(manifest)[0]
    with np.load(root / record.relative_path) as data:
        result = predictor.predict(data["csi"], data["timestamps_ns"])
    assert 0 <= result["predicted_class"] < 24
    assert len(result["top_k"]) == 3
    assert len(result["model_sha256"]) == 64
    assert result["top_k"][0]["class_name"].startswith("activity_")


def test_evaluation_rejects_mutated_manifest(model_artifact):
    root, manifest, output, _ = model_artifact
    changed = root / "modified.jsonl"
    rows = manifest.read_text().splitlines()
    first = json.loads(rows[0]); first["domain_id"] = "changed_environment"
    rows[0] = json.dumps(first)
    changed.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="Manifest changed"):
        evaluate(output / "best_model.pt", changed, root, output / "bad.json")


def test_loso_partial_subject_fold(tmp_path):
    from crossguard_fi.data.synthetic import generate_fixture
    from crossguard_fi.training.loso import run_loso
    root = tmp_path / "loso-data"
    manifest = generate_fixture(root, subjects=3, samples=100)
    report = run_loso(manifest, root, tmp_path / "loso-results",
                      TrainConfig(model="mlp", epochs=1, hidden_dim=16,
                                  batch_size=24), max_folds=1)
    assert report["folds_completed"] == 1
    assert report["complete_loso"] is False
    assert report["folds"][0]["test_subject"] != report["folds"][0]["val_subject"]


def test_evaluation_rejects_recording_byte_mutation(model_artifact):
    root, manifest, output, _ = model_artifact
    records = load_manifest(manifest)
    target = root / records[0].relative_path
    original = target.read_bytes()
    try:
        target.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="Recording bytes changed"):
            evaluate(output / "best_model.pt", manifest, root, output / "tampered.json")
    finally:
        target.write_bytes(original)
