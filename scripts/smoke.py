"""One-command deterministic synthetic training/validation/test and prediction smoke test."""
import json
from pathlib import Path
import tempfile

import numpy as np
from crossguard_fi.config import TrainConfig
from crossguard_fi.data.manifest import load_manifest
from crossguard_fi.data.synthetic import generate_fixture
from crossguard_fi.serving.predictor import Predictor
from crossguard_fi.training.engine import train, evaluate


def main():
    with tempfile.TemporaryDirectory(prefix="crossguard-fi-smoke-") as folder:
        root = Path(folder)
        manifest = generate_fixture(root, subjects=3, samples=100)
        report = train(manifest, root, root / "run", TrainConfig(
            model="mlp", hidden_dim=16, epochs=1, batch_size=16, device="cpu"))
        checkpoint = root / "run" / "best_model.pt"
        test_report = evaluate(checkpoint, manifest, root, root / "run" / "sealed_test.json")
        with np.load(root / load_manifest(manifest)[0].relative_path) as data:
            prediction = Predictor(checkpoint).predict(data["csi"], data["timestamps_ns"])
        assert len(prediction["top_k"]) == 3
        assert test_report["samples"] == 24
        print(json.dumps({
            "synthetic_only": True,
            "training_validation_completed": report["best_epoch"] == 1,
            "sealed_test_windows": test_report["samples"],
            "inference_class_valid": 0 <= prediction["predicted_class"] < 24,
            "disclaimer": "Not a field result or claimed CSI recognition accuracy",
        }, indent=2))


if __name__ == "__main__":
    main()
