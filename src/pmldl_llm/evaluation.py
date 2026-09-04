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


def evaluate_experiment_probabilities(
    y_true: np.ndarray,
    original_probabilities: np.ndarray,
    swapped_back_probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute the complete scalar contract after A/B symmetry averaging.

    ``swapped_back_probabilities`` must already be mapped to the original
    A/B/tie column order.
    """
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("Targets must be a non-empty one-dimensional array.")
    integer_targets = y_true.astype(np.int64)
    if not np.array_equal(y_true, integer_targets) or not np.isin(
        integer_targets, [0, 1, 2]
    ).all():
        raise ValueError("Targets must contain only class indices 0, 1, and 2.")
    original = normalize_probabilities(original_probabilities)
    swapped_back = normalize_probabilities(swapped_back_probabilities)
    if len(y_true) != len(original) or original.shape != swapped_back.shape:
        raise ValueError("Targets, original predictions, and swapped predictions differ.")
    averaged = normalize_probabilities(0.5 * (original + swapped_back))
    evaluation = evaluate_probabilities(integer_targets, averaged)
    one_hot = np.eye(3, dtype=np.float64)[integer_targets]
    return {
        "log_loss": float(evaluation["log_loss"]),
        "accuracy": float(evaluation["accuracy"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "ece_15": float(evaluation["ece_15"]),
        "brier_score": float(np.square(averaged - one_hot).sum(axis=1).mean()),
        "swap_error_l1": float(
            np.abs(original - swapped_back).sum(axis=1).mean()
        ),
    }
