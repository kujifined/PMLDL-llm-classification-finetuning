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
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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
from pmldl_llm.text import flattened_text_columns


def ordered_predict_proba(model: SGDClassifier, features: sparse.csr_matrix) -> np.ndarray:
    raw = model.predict_proba(features)
    ordered = np.zeros((features.shape[0], 3), dtype=np.float64)
    ordered[:, model.classes_.astype(int)] = raw
    return normalize_probabilities(ordered)


def vectorize_parts(
    vectorizer: TfidfVectorizer,
    frame: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    prompts, responses_a, responses_b = flattened_text_columns(frame)
    return (
        vectorizer.transform(prompts).tocsr(),
        vectorizer.transform(responses_a).tocsr(),
        vectorizer.transform(responses_b).tocsr(),
    )


def raw_dense_features(
    frame: pd.DataFrame,
    prompt_matrix: sparse.csr_matrix,
    response_a_matrix: sparse.csr_matrix,
    response_b_matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, list[str]]:
    bias, names = build_bias_features(frame)
    similarity_a = np.asarray(
        prompt_matrix.multiply(response_a_matrix).sum(axis=1)
    ).ravel()
    similarity_b = np.asarray(
        prompt_matrix.multiply(response_b_matrix).sum(axis=1)
    ).ravel()
    similarities = np.column_stack(
        [
            similarity_a,
            similarity_b,
            similarity_a - similarity_b,
            np.abs(similarity_a - similarity_b),
        ]
    )
    return (
        np.column_stack([bias, similarities]),
        names
        + [
            "prompt_response_a_cosine",
            "prompt_response_b_cosine",
            "prompt_similarity_difference",
            "absolute_prompt_similarity_difference",
        ],
    )


def compose_features(
    response_a_matrix: sparse.csr_matrix,
    response_b_matrix: sparse.csr_matrix,
    scaled_dense: np.ndarray,
) -> sparse.csr_matrix:
    difference = response_a_matrix - response_b_matrix
    absolute_difference = abs(difference)
    return sparse.hstack(
        [
            difference,
            absolute_difference,
            sparse.csr_matrix(scaled_dense, dtype=np.float32),
        ],
        format="csr",
        dtype=np.float32,
    )


def swapped_compose_features(
    response_a_matrix: sparse.csr_matrix,
    response_b_matrix: sparse.csr_matrix,
    scaled_swapped_dense: np.ndarray,
) -> sparse.csr_matrix:
    difference = response_b_matrix - response_a_matrix
    absolute_difference = abs(difference)
    return sparse.hstack(
        [
            difference,
            absolute_difference,
            sparse.csr_matrix(scaled_swapped_dense, dtype=np.float32),
        ],
        format="csr",
        dtype=np.float32,
    )


def build_model(config: dict[str, object], alpha: float) -> SGDClassifier:
    tolerance = config.get("tolerance")
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=int(config["max_iter"]),
        tol=None if tolerance is None else float(tolerance),
        shuffle=True,
        random_state=int(config["seed"]),
        average=True,
    )


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sparse_baseline.json",
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
        default=PROJECT_ROOT / "artifacts" / "sparse_baseline",
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
    fold_values = folds["fold"].to_numpy()
    training_folds = list(roles.training_folds)
    is_training = np.isin(fold_values, training_folds)
    is_validation = fold_values == roles.validation_fold
    if not is_training.any() or not is_validation.any():
        raise RuntimeError("Training and validation selections must be non-empty.")
    y = target_indices(train)
    y_train = y[is_training]
    y_validation = y[is_validation]
    training_frame = train.loc[is_training].reset_index(drop=True)
    validation_frame = train.loc[is_validation].reset_index(drop=True)

    train_prompt, train_a, train_b = flattened_text_columns(training_frame)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(int(config["ngram_min"]), int(config["ngram_max"])),
        min_df=int(config["min_df"]),
        max_df=float(config["max_df"]),
        max_features=int(config["max_features"]),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vectorizer.fit(train_prompt + train_a + train_b)
    train_prompt_matrix = vectorizer.transform(train_prompt).tocsr()
    train_a_matrix = vectorizer.transform(train_a).tocsr()
    train_b_matrix = vectorizer.transform(train_b).tocsr()

    train_dense, dense_names = raw_dense_features(
        training_frame,
        train_prompt_matrix,
        train_a_matrix,
        train_b_matrix,
    )
    swapped_train_dense, swapped_dense_names = raw_dense_features(
        swapped_frame(training_frame),
        train_prompt_matrix,
        train_b_matrix,
        train_a_matrix,
    )
    if swapped_dense_names != dense_names:
        raise RuntimeError("Dense feature order changed after swapping.")
    scaler = StandardScaler()
    scaler.fit(np.vstack([train_dense, swapped_train_dense]))
    train_features = compose_features(
        train_a_matrix,
        train_b_matrix,
        scaler.transform(train_dense),
    )
    swapped_train_features = swapped_compose_features(
        train_a_matrix,
        train_b_matrix,
        scaler.transform(swapped_train_dense),
    )
    augmented_features = sparse.vstack(
        [train_features, swapped_train_features],
        format="csr",
    )
    augmented_targets = np.concatenate([y_train, swap_target_indices(y_train)])

    validation_prompt_matrix, validation_a_matrix, validation_b_matrix = (
        vectorize_parts(vectorizer, validation_frame)
    )
    validation_dense, _ = raw_dense_features(
        validation_frame,
        validation_prompt_matrix,
        validation_a_matrix,
        validation_b_matrix,
    )
    swapped_validation_dense, _ = raw_dense_features(
        swapped_frame(validation_frame),
        validation_prompt_matrix,
        validation_b_matrix,
        validation_a_matrix,
    )
    validation_features = compose_features(
        validation_a_matrix,
        validation_b_matrix,
        scaler.transform(validation_dense),
    )
    swapped_validation_features = swapped_compose_features(
        validation_a_matrix,
        validation_b_matrix,
        scaler.transform(swapped_validation_dense),
    )

    candidate_metrics = {}
    candidate_models = {}
    candidate_probabilities = {}
    for alpha_value in config["alpha_candidates"]:
        alpha = float(alpha_value)
        model = build_model(config, alpha)
        model.fit(augmented_features, augmented_targets)
        original = ordered_predict_proba(model, validation_features)
        swapped_back = swap_probability_columns(
            ordered_predict_proba(model, swapped_validation_features)
        )
        probability = normalize_probabilities(0.5 * (original + swapped_back))
        key = f"alpha={alpha:g}"
        candidate_metrics[key] = {
            **evaluate_probabilities(y_validation, probability),
            "raw_symmetry_l1": float(
                np.abs(original - swapped_back).sum(axis=1).mean()
            ),
        }
        candidate_models[key] = model
        candidate_probabilities[key] = probability

    best_key = min(
        candidate_metrics,
        key=lambda key: candidate_metrics[key]["log_loss"],
    )
    best_model = candidate_models[best_key]
    validation_probability = candidate_probabilities[best_key]

    test_prompt_matrix, test_a_matrix, test_b_matrix = vectorize_parts(
        vectorizer, test
    )
    test_dense, _ = raw_dense_features(
        test,
        test_prompt_matrix,
        test_a_matrix,
        test_b_matrix,
    )
    swapped_test_dense, _ = raw_dense_features(
        swapped_frame(test),
        test_prompt_matrix,
        test_b_matrix,
        test_a_matrix,
    )
    test_features = compose_features(
        test_a_matrix,
        test_b_matrix,
        scaler.transform(test_dense),
    )
    swapped_test_features = swapped_compose_features(
        test_a_matrix,
        test_b_matrix,
        scaler.transform(swapped_test_dense),
    )
    test_original = ordered_predict_proba(best_model, test_features)
    test_swapped_back = swap_probability_columns(
        ordered_predict_proba(best_model, swapped_test_features)
    )
    test_probability = normalize_probabilities(
        0.5 * (test_original + test_swapped_back)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_smoke_path = write_submission(
        test["id"].to_numpy(),
        test_probability,
        args.output_dir / "test_schema_predictions.csv",
    )
    pd.DataFrame(
        {
            "id": validation_frame["id"].to_numpy(),
            "target": y_validation,
            TARGET_COLUMNS[0]: validation_probability[:, 0],
            TARGET_COLUMNS[1]: validation_probability[:, 1],
            TARGET_COLUMNS[2]: validation_probability[:, 2],
        }
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "protocol": {
            "group_key": "sha256(normalized_prompt)",
            "fold_assignment": path_label(args.folds, PROJECT_ROOT),
            "n_splits": roles.n_splits,
            "training_folds": training_folds,
            "validation_fold": roles.validation_fold,
            "calibration_fold": roles.calibration_fold,
            "final_holdout_fold": roles.final_holdout_fold,
            "split_seed": roles.seed,
            "model_seed": int(config["seed"]),
            "train_rows": int(len(training_frame)),
            "validation_rows": int(len(validation_frame)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "matrix_columns": int(train_features.shape[1]),
            "matrix_nonzero": int(train_features.nnz),
            "a_b_swap_augmentation": True,
            "a_b_swap_inference_average": True,
            "optimization": (
                f"fixed {int(config['max_iter'])} SGD epochs"
                if config.get("tolerance") is None
                else f"up to {int(config['max_iter'])} SGD epochs"
            ),
            "evaluation_checkpoint": "evaluation_model.joblib",
            "evaluation_checkpoint_scope": "folds 0-6 only",
        },
        "candidates": candidate_metrics,
        "best_candidate": best_key,
        "best_validation": candidate_metrics[best_key],
        "test_schema_smoke": {
            "rows": int(len(test)),
            "prediction_file": schema_smoke_path.name,
            "warning": (
                "Three-row schema check from the evaluation checkpoint; "
                "this is not a Kaggle submission."
            ),
        },
        "provenance": build_provenance(
            PROJECT_ROOT,
            __file__,
            [args.config, args.split_config],
            args.folds,
            dataset_hashes,
        ),
        "runtime_seconds": float(time.perf_counter() - started_at),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "scaler": scaler,
            "dense_feature_names": dense_names,
            "target_columns": TARGET_COLUMNS,
            "experiment_config": config,
            "split_config": split_config,
            "best_candidate": best_key,
            "training_folds": training_folds,
        },
        args.output_dir / "evaluation_model.joblib",
        compress=3,
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved sparse baseline artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
