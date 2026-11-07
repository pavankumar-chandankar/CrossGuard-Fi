"""End-user interface contract tests without AWS/network dependencies."""
import json
from pathlib import Path

from crossguard_fi.cli import build_parser, main


def test_command_surface():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert {"generate-fixture", "validate", "train", "evaluate", "predict", "benchmark", "loso"} <= set(commands)


def test_generated_fixture_cli_and_validation(tmp_path, capsys):
    root = tmp_path / "demo"
    main(["generate-fixture", "--output", str(root), "--subjects", "3", "--samples", "100"])
    generated = json.loads(capsys.readouterr().out)
    assert generated["synthetic_only"] is True
    main(["validate", "--manifest", str(root / "manifest.jsonl"),
          "--data-root", str(root)])
    check = json.loads(capsys.readouterr().out)
    assert check["recordings"] == 72
    assert check["total_samples"] == 7200


def test_train_evaluate_predict_cli(tmp_path, capsys):
    root = tmp_path / "dataset"
    main(["generate-fixture", "--output", str(root), "--subjects", "3", "--samples", "100"])
    capsys.readouterr()
    manifest = root / "manifest.jsonl"
    run = tmp_path / "run"
    main(["train", "--manifest", str(manifest), "--data-root", str(root),
          "--output", str(run), "--model", "mlp", "--epochs", "1",
          "--hidden-dim", "16", "--batch-size", "24", "--device", "cpu"])
    train_report = json.loads(capsys.readouterr().out)
    assert train_report["best_epoch"] == 1
    main(["evaluate", "--manifest", str(manifest), "--data-root", str(root),
          "--checkpoint", str(run / "best_model.pt"),
          "--output", str(run / "evaluation.json")])
    result = json.loads(capsys.readouterr().out)
    assert result["artifact_type"] == "sealed_subject_test"
    main(["predict", "--checkpoint", str(run / "best_model.pt"),
          "--recording", str(root / "recordings/s00_c00_t00.npz")])
    result = json.loads(capsys.readouterr().out)
    assert len(result["top_k"]) == 3
