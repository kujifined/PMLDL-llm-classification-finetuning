#!/usr/bin/env python3
"""Recompute predictions from saved checkpoints and compare every saved row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
    swapped_frame,
    target_indices,
    verify_competition_data_dir,
    verify_checksum_manifest,
)
from pmldl_llm.evaluation import normalize_probabilities
from pmldl_llm.split import load_frozen_folds
from train_bias_baseline import symmetry_average as bias_symmetry_average
from train_sparse_baseline import (
    compose_features,
    ordered_predict_proba,
    raw_dense_features,
    swapped_compose_features,
    vectorize_parts,
)


def assert_prediction_file(
    path: Path,
    ids: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray | None = None,
) -> None:
    saved = pd.read_csv(path)
    if not np.array_equal(saved["id"].to_numpy(), ids):
        raise AssertionError(f"ID/order mismatch in {path}.")
    if targets is not None:
        if "target" not in saved or not np.array_equal(
            saved["target"].to_numpy(dtype=np.int64), targets
        ):
            raise AssertionError(f"Target mismatch in {path}.")
    np.testing.assert_allclose(
        saved[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64),
        probabilities,
        rtol=0.0,
        atol=5e-15,
        err_msg=f"Probability mismatch in {path}",
    )


def sparse_probabilities(bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    prompt_matrix, response_a_matrix, response_b_matrix = vectorize_parts(
        vectorizer, frame
    )
    dense, _ = raw_dense_features(
        frame, prompt_matrix, response_a_matrix, response_b_matrix
    )
    swapped_dense, _ = raw_dense_features(
        swapped_frame(frame), prompt_matrix, response_b_matrix, response_a_matrix
    )
    original_features = compose_features(
        response_a_matrix, response_b_matrix, scaler.transform(dense)
    )
    swapped_features = swapped_compose_features(
        response_a_matrix, response_b_matrix, scaler.transform(swapped_dense)
    )
    original = ordered_predict_proba(model, original_features)
    swapped_back = swap_probability_columns(
        ordered_predict_proba(model, swapped_features)
    )
    return normalize_probabilities(0.5 * (original + swapped_back))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that saved checkpoints exactly reproduce prediction artifacts."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines",
    )
    args = parser.parse_args()

    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    train, test = load_competition_data(args.data_dir)
    folds = load_frozen_folds(
        train,
        args.folds,
        args.split_config,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )
    mask = folds["fold"].to_numpy() == roles.validation_fold
    validation = train.loc[mask].reset_index(drop=True)
    validation_targets = target_indices(train)[mask]

    bias_dir = args.artifacts_dir / "bias_baseline"
    bias_evaluation = joblib.load(bias_dir / "evaluation_model.joblib")
    bias_validation_probability, _ = bias_symmetry_average(
        bias_evaluation["model"], validation
    )
    assert_prediction_file(
        bias_dir / "validation_predictions.csv",
        validation["id"].to_numpy(),
        bias_validation_probability,
        validation_targets,
    )
    bias_full = joblib.load(bias_dir / "full_model.joblib")
    bias_test_probability, _ = bias_symmetry_average(bias_full["model"], test)
    assert_prediction_file(
        bias_dir / "test_schema_predictions.csv",
        test["id"].to_numpy(),
        bias_test_probability,
    )

    sparse_dir = args.artifacts_dir / "sparse_baseline"
    sparse_evaluation = joblib.load(sparse_dir / "evaluation_model.joblib")
    sparse_validation_probability = sparse_probabilities(
        sparse_evaluation, validation
    )
    assert_prediction_file(
        sparse_dir / "validation_predictions.csv",
        validation["id"].to_numpy(),
        sparse_validation_probability,
        validation_targets,
    )
    sparse_test_probability = sparse_probabilities(sparse_evaluation, test)
    assert_prediction_file(
        sparse_dir / "test_schema_predictions.csv",
        test["id"].to_numpy(),
        sparse_test_probability,
    )

    blend_dir = args.artifacts_dir / "baseline_blend"
    blend_metrics = json.loads(
        (args.results_dir / "baseline_blend" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    structural_weight = float(blend_metrics["best_structural_weight"])
    sparse_weight = float(blend_metrics["best_sparse_weight"])
    if not np.isclose(structural_weight + sparse_weight, 1.0, atol=1e-15):
        raise AssertionError("Saved blend weights do not sum to one.")
    blend_probability = normalize_probabilities(
        structural_weight * bias_validation_probability
        + sparse_weight * sparse_validation_probability
    )
    assert_prediction_file(
        blend_dir / "validation_predictions.csv",
        validation["id"].to_numpy(),
        blend_probability,
        validation_targets,
    )
    print(
        "OK: structural, sparse, and blend artifacts reproduce all 5,746 "
        "validation rows; both schema prediction files reproduce all 3 rows."
    )


if __name__ == "__main__":
    main()
