from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FoldRoles:
    n_splits: int
    seed: int
    training_folds: tuple[int, ...]
    validation_fold: int
    calibration_fold: int
    final_holdout_fold: int


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return value


def validate_fold_roles(config: Mapping[str, Any]) -> FoldRoles:
    """Validate that every fold has exactly one pre-declared role."""
    required = {
        "n_splits",
        "seed",
        "training_folds",
        "validation_fold",
        "calibration_fold",
        "final_holdout_fold",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Split config is missing fields: {sorted(missing)}")

    n_splits = _strict_int(config["n_splits"], "n_splits")
    seed = _strict_int(config["seed"], "seed")
    if n_splits < 4:
        raise ValueError("n_splits must be at least 4 for train/selection/calibration/holdout.")

    raw_training = config["training_folds"]
    if not isinstance(raw_training, list) or not raw_training:
        raise ValueError("training_folds must be a non-empty list.")
    training_folds = tuple(
        _strict_int(value, f"training_folds[{index}]")
        for index, value in enumerate(raw_training)
    )
    if len(set(training_folds)) != len(training_folds):
        raise ValueError("training_folds contains duplicate folds.")

    validation_fold = _strict_int(config["validation_fold"], "validation_fold")
    calibration_fold = _strict_int(config["calibration_fold"], "calibration_fold")
    final_holdout_fold = _strict_int(
        config["final_holdout_fold"], "final_holdout_fold"
    )
    all_roles = (
        *training_folds,
        validation_fold,
        calibration_fold,
        final_holdout_fold,
    )
    if any(fold < 0 or fold >= n_splits for fold in all_roles):
        raise ValueError(f"Every fold role must be in [0, {n_splits - 1}].")
    if len(set(all_roles)) != len(all_roles):
        raise ValueError("Train, selection, calibration, and holdout folds must be disjoint.")
    expected = set(range(n_splits))
    if set(all_roles) != expected:
        missing_folds = sorted(expected.difference(all_roles))
        raise ValueError(f"Every fold must have exactly one role; unassigned: {missing_folds}.")

    return FoldRoles(
        n_splits=n_splits,
        seed=seed,
        training_folds=training_folds,
        validation_fold=validation_fold,
        calibration_fold=calibration_fold,
        final_holdout_fold=final_holdout_fold,
    )
