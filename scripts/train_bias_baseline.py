#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.config import validate_fold_roles
from pmldl_llm.constants import (
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
    TARGET_COLUMNS,
)
from pmldl_llm.data import (
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    swapped_frame,
    target_indices,
    verify_competition_data_dir,
    verify_checksum_manifest,
)
from pmldl_llm.evaluation import evaluate_probabilities, normalize_probabilities
from pmldl_llm.features import build_bias_features
from pmldl_llm.provenance import build_provenance, path_label
from pmldl_llm.split import load_frozen_folds
from pmldl_llm.submission import write_submission


def ordered_predict_proba(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["logisticregression"].classes_.astype(int)
    ordered = np.zeros((len(features), 3), dtype=np.float64)
    ordered[:, classes] = probabilities
    return normalize_probabilities(ordered)


def symmetry_average(
    model: object,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    features, _ = build_bias_features(frame)
    swapped_features, _ = build_bias_features(swapped_frame(frame))
    original = ordered_predict_proba(model, features)
    swapped_back = swap_probability_columns(
        ordered_predict_proba(model, swapped_features)
    )
    raw_error = float(np.abs(original - swapped_back).sum(axis=1).mean())
    return normalize_probabilities(0.5 * (original + swapped_back)), raw_error


def build_model(config: dict[str, object]) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config["logistic_c"]),
            max_iter=int(config["max_iter"]),
            solver="lbfgs",
            random_state=int(config["seed"]),
        ),
    )


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "bias_baseline.json",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "bias_baseline",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "baselines"
            / "bias_baseline"
            / "evaluation.json"
        ),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    train, test = load_competition_data(args.data_dir)
    folds = load_frozen_folds(
        train,
        args.folds,
        args.split_config,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )
    training_folds = list(roles.training_folds)
    validation_fold = roles.validation_fold
    fold_values = folds["fold"].to_numpy()
    is_training = np.isin(fold_values, training_folds)
    is_validation = fold_values == validation_fold
    if (is_training & is_validation).any():
        raise RuntimeError("Training and validation folds overlap.")
    if not is_training.any() or not is_validation.any():
        raise RuntimeError("Training and validation selections must be non-empty.")
    y = target_indices(train)

    training_frame = train.loc[is_training].reset_index(drop=True)
    validation_frame = train.loc[is_validation].reset_index(drop=True)
    y_train = y[is_training]
    y_validation = y[is_validation]

    train_features, feature_names = build_bias_features(training_frame)
    swapped_train_features, swapped_names = build_bias_features(
        swapped_frame(training_frame)
    )
    if swapped_names != feature_names:
        raise RuntimeError("Feature order changed after A/B swapping.")
    augmented_features = np.vstack([train_features, swapped_train_features])
    augmented_targets = np.concatenate([y_train, swap_target_indices(y_train)])

    uniform = np.full((len(y_validation), 3), 1.0 / 3.0)
    train_prior = np.bincount(y_train, minlength=3).astype(np.float64)
    train_prior /= train_prior.sum()
    prior = np.tile(train_prior, (len(y_validation), 1))

    model = build_model(config)
    model.fit(augmented_features, augmented_targets)
    validation_probability, raw_symmetry_error = symmetry_average(
        model, validation_frame
    )

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "protocol": {
            "group_key": "sha256(normalized_prompt)",
            "fold_assignment": path_label(args.folds, PROJECT_ROOT),
            "n_splits": roles.n_splits,
            "training_folds": training_folds,
            "validation_fold": validation_fold,
            "calibration_fold": roles.calibration_fold,
            "final_holdout_fold": roles.final_holdout_fold,
            "split_seed": roles.seed,
            "model_seed": int(config["seed"]),
            "train_rows": int(len(training_frame)),
            "validation_rows": int(len(validation_frame)),
            "feature_count": len(feature_names),
            "a_b_swap_augmentation": True,
            "a_b_swap_inference_average": True,
            "evaluation_checkpoint": "evaluation_model.joblib",
            "evaluation_checkpoint_scope": "folds 0-6 only",
        },
        "uniform": evaluate_probabilities(y_validation, uniform),
        "train_prior": evaluate_probabilities(y_validation, prior),
        "bias_logistic": evaluate_probabilities(
            y_validation, validation_probability
        ),
        "raw_symmetry_l1": raw_symmetry_error,
        "symmetry_l1_after_average": 0.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_output = pd.DataFrame(
        {
            "id": validation_frame["id"].to_numpy(),
            "target": y_validation,
            TARGET_COLUMNS[0]: validation_probability[:, 0],
            TARGET_COLUMNS[1]: validation_probability[:, 1],
            TARGET_COLUMNS[2]: validation_probability[:, 2],
        }
    )
    validation_output.to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )

    full_features, full_names = build_bias_features(train)
    full_swapped_features, _ = build_bias_features(swapped_frame(train))
    if full_names != feature_names:
        raise RuntimeError("Feature order changed between split and full data.")
    final_model = build_model(config)
    final_model.fit(
        np.vstack([full_features, full_swapped_features]),
        np.concatenate([y, swap_target_indices(y)]),
    )
    test_probability, test_raw_symmetry_error = symmetry_average(final_model, test)
    schema_smoke_path = write_submission(
        test["id"].to_numpy(),
        test_probability,
        args.output_dir / "test_schema_predictions.csv",
    )
    metrics["test_schema_smoke"] = {
        "rows": int(len(test)),
        "raw_symmetry_l1": test_raw_symmetry_error,
        "prediction_file": schema_smoke_path.name,
        "warning": "Three-row schema check only; this is not a Kaggle submission.",
        "checkpoint": "full_model.joblib",
        "checkpoint_scope": "all labeled training rows",
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_columns": TARGET_COLUMNS,
            "experiment_config": config,
            "split_config": split_config,
            "training_folds": training_folds,
        },
        args.output_dir / "evaluation_model.joblib",
    )
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "target_columns": TARGET_COLUMNS,
            "experiment_config": config,
            "split_config": split_config,
            "training_scope": "all labeled rows; schema smoke only",
        },
        args.output_dir / "full_model.joblib",
    )
    coefficients = final_model.named_steps["logisticregression"].coef_
    coefficient_rows = []
    for class_index, row in zip(
        final_model.named_steps["logisticregression"].classes_,
        coefficients,
    ):
        for feature_name, coefficient in zip(feature_names, row):
            coefficient_rows.append(
                {
                    "class_index": int(class_index),
                    "target": TARGET_COLUMNS[int(class_index)],
                    "feature": feature_name,
                    "standardized_coefficient": float(coefficient),
                }
            )
    pd.DataFrame(coefficient_rows).to_csv(
        args.output_dir / "feature_coefficients.csv", index=False
    )
    metrics["provenance"] = build_provenance(
        PROJECT_ROOT,
        __file__,
        [args.config, args.split_config],
        args.folds,
        dataset_hashes,
    )
    metrics["runtime_seconds"] = float(time.perf_counter() - started_at)
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_output.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved baseline artifacts to {args.output_dir}")
    print(f"Saved baseline evaluation to {args.evaluation_output}")


if __name__ == "__main__":
    main()
