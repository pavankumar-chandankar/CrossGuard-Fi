"""Explicit class-complete metrics, calibration and confusion matrices.

All macro metrics use the fixed 24-class universe, including absent test classes.
"""
import numpy as np
from crossguard_fi.config import CONTRACT


def classification_metrics(actual: np.ndarray, probabilities: np.ndarray,
                           classes: int = CONTRACT.classes, bins: int = 10) -> dict:
    actual = np.asarray(actual, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or len(actual) == 0:
        raise ValueError("At least one label is required")
    if probabilities.shape != (len(actual), classes):
        raise ValueError("Expected [N, classes] probability matrix")
    if np.any(actual < 0) or np.any(actual >= classes):
        raise ValueError("Target label out of range")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Invalid model probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Probabilities must sum to one")
    predicted = probabilities.argmax(axis=1)
    matrix = np.bincount(actual * classes + predicted,
                         minlength=classes * classes).reshape(classes, classes)
    true_count = matrix.sum(axis=1).astype(float)
    predicted_count = matrix.sum(axis=0).astype(float)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1],
                            np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)
        if np.any(selected):
            ece += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    nll = -np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-9, 1)).mean()
    return {
        "samples": int(len(actual)), "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy_present_classes": float(recall[true_count > 0].mean()),
        "ece_10_bins": ece, "negative_log_likelihood": float(nll),
        "classes_present": int(np.count_nonzero(true_count)),
        "per_class": [{"class_id": i, "support": int(true_count[i]),
                        "precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i])} for i in range(classes)],
        "confusion_matrix": matrix.tolist(),
    }
