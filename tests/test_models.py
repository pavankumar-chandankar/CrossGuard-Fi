import numpy as np
import pytest
import torch

from crossguard_fi.models.networks import build_model
from crossguard_fi.training.metrics import classification_metrics


@pytest.mark.parametrize("name", ["tcn", "mlp"])
def test_model_shapes_and_gradients(name):
    torch.manual_seed(5)
    model = build_model(name, hidden_dim=16)
    x = torch.randn(2, 100, 270)
    scores = model(x)
    assert tuple(scores.shape) == (2, 24)
    scores.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_model_rejects_wrong_dimensionality():
    model = build_model("mlp", hidden_dim=16)
    with pytest.raises(ValueError):
        model(torch.zeros((2, 99, 270)))


def test_metrics_and_confusion():
    actual = np.array([0, 0, 1, 1])
    p = np.zeros((4, 24))
    p[[0, 1, 2], [0, 0, 1]] = 1
    p[3, 0] = 1
    result = classification_metrics(actual, p)
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"][1][0] == 1
    assert result["classes_present"] == 2
    assert result["macro_f1"] < 0.1  # fixed 24-class macro average, not 2-class


def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError):
        classification_metrics(np.array([0]), np.zeros((1, 24)))
