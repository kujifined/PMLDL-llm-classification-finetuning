from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def normalize_probabilities(
    probabilities: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Expected probabilities with shape (n_rows, 3).")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    probabilities = np.clip(probabilities, epsilon, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    probabilities = normalize_probabilities(probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        include = (confidence > lower) & (confidence <= upper)
        if include.any():
            result += include.mean() * abs(
                correctness[include].mean() - confidence[include].mean()
            )
    return float(result)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    probabilities = normalize_probabilities(probabilities)
    predictions = probabilities.argmax(axis=1)
    class_rates = np.bincount(predictions, minlength=3).astype(float)
    class_rates /= max(len(predictions), 1)
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "ece_15": expected_calibration_error(y_true, probabilities, n_bins=15),
        "predicted_class_rates": class_rates.tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1, 2]
        ).tolist(),
    }

