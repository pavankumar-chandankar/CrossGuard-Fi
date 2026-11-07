import os
import numpy as np
from fastapi.testclient import TestClient
import pytest

from crossguard_fi.serving.api import create_app
from crossguard_fi.config import CONTRACT


def test_untrained_app_exposes_liveness_not_readiness(monkeypatch):
    monkeypatch.delenv("CGF_MODEL_PATH", raising=False)
    monkeypatch.delenv("CGF_API_KEY", raising=False)
    monkeypatch.setenv("CGF_ENV", "development")
    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/v1/predict", json={"samples": [], "timestamps_ns": []}).status_code == 422


def test_production_requires_auth(monkeypatch):
    monkeypatch.setenv("CGF_ENV", "production")
    monkeypatch.delenv("CGF_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="requires CGF_API_KEY"):
        create_app()


def test_authenticated_api_and_model(monkeypatch, tmp_path):
    from crossguard_fi.data.synthetic import generate_fixture
    from crossguard_fi.training.engine import train
    from crossguard_fi.config import TrainConfig
    from crossguard_fi.serving.predictor import Predictor

    root = tmp_path / "data"; manifest = generate_fixture(root, subjects=3, samples=100)
    checkpoint_root = tmp_path / "run"
    train(manifest, root, checkpoint_root, TrainConfig(model="mlp", epochs=1,
          batch_size=32, hidden_dim=16))
    predictor = Predictor(checkpoint_root / "best_model.pt")
    monkeypatch.setenv("CGF_API_KEY", "test-key")
    monkeypatch.setenv("CGF_ENV", "production")
    client = TestClient(create_app(predictor))
    sample = np.zeros((100, 270), dtype=float).tolist()
    request = {"samples": sample, "timestamps_ns": (np.arange(100) * 20_000_000).tolist()}
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/predict", json=request).status_code == 401
    resp = client.post("/v1/predict", json=request, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["top_k"]) == 3
    request["samples"][0][0] = 10_000_000
    assert client.post("/v1/predict", json=request,
                       headers={"X-API-Key": "test-key"}).status_code == 422
