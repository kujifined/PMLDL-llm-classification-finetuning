#!/usr/bin/env python3
"""Sequential word/character TF-IDF sweep for the Sprint 2 sparse task."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.constants import (  # noqa: E402
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
    TARGET_COLUMNS,
)
from pmldl_llm.data import (  # noqa: E402
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    swapped_frame,
    target_indices,
    verify_checksum_manifest,
    verify_competition_data_dir,
)
from pmldl_llm.evaluation import (  # noqa: E402
    evaluate_experiment_probabilities,
    normalize_probabilities,
)
from pmldl_llm.experiment import ExperimentRun  # noqa: E402
from pmldl_llm.features import build_bias_features  # noqa: E402
from pmldl_llm.split import load_frozen_folds  # noqa: E402
from pmldl_llm.submission import write_submission  # noqa: E402
from pmldl_llm.text import flattened_text_columns  # noqa: E402
from pmldl_llm.provenance import path_label  # noqa: E402
from pmldl_llm.config import validate_fold_roles  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    alpha: float
    char_ngram_range: tuple[int, int] = (3, 5)
    char_max_features: int = 50000

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "alpha": self.alpha,
            "char_ngram_range": list(self.char_ngram_range),
            "char_max_features": self.char_max_features,
        }


def ordered_predict_proba(model: Any, features: sparse.csr_matrix) -> np.ndarray:
    raw = model.predict_proba(features)
    ordered = np.zeros((features.shape[0], 3), dtype=np.float64)
    ordered[:, model.classes_.astype(int)] = raw
    return normalize_probabilities(ordered)


def make_vectorizer(
    *, analyzer: str, ngram_range: tuple[int, int], max_features: int, min_df: int
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def vectorizers_for(candidate: Candidate) -> list[TfidfVectorizer]:
    if candidate.analyzer == "word":
        return [
            make_vectorizer(
                analyzer="word",
                ngram_range=candidate.ngram_range,
                max_features=candidate.max_features,
                min_df=candidate.min_df,
            )
        ]
    if candidate.analyzer == "char":
        return [
            make_vectorizer(
                analyzer="char",
                ngram_range=candidate.char_ngram_range,
                max_features=candidate.char_max_features,
                min_df=candidate.min_df,
            )
        ]
    if candidate.analyzer == "word_char":
        return [
            make_vectorizer(
                analyzer="word",
                ngram_range=candidate.ngram_range,
                max_features=candidate.max_features,
                min_df=candidate.min_df,
            ),
            make_vectorizer(
                analyzer="char",
                ngram_range=candidate.char_ngram_range,
                max_features=candidate.char_max_features,
                min_df=candidate.min_df,
            ),
        ]
    raise ValueError(f"Unknown analyzer: {candidate.analyzer}")


def vectorize_parts(
    vectorizer: TfidfVectorizer, frame: pd.DataFrame
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


def swapped_dense_features(dense: np.ndarray, names: list[str]) -> np.ndarray:
    """Return dense features for the same rows after swapping response A/B."""
    name_to_index = {name: index for index, name in enumerate(names)}
    swapped = dense.copy()
    for name, index in name_to_index.items():
        if name.startswith("a_"):
            swapped[:, index] = dense[:, name_to_index["b_" + name[2:]]]
        elif name.startswith("b_"):
            swapped[:, index] = dense[:, name_to_index["a_" + name[2:]]]
        elif name.startswith("diff_") or name.startswith("log_ratio_"):
            swapped[:, index] = -dense[:, index]
        elif name == "prompt_response_a_cosine":
            swapped[:, index] = dense[:, name_to_index["prompt_response_b_cosine"]]
        elif name == "prompt_response_b_cosine":
            swapped[:, index] = dense[:, name_to_index["prompt_response_a_cosine"]]
        elif name == "prompt_similarity_difference":
            swapped[:, index] = -dense[:, index]
    return swapped


def compose_features(
    response_a_matrices: list[sparse.csr_matrix],
    response_b_matrices: list[sparse.csr_matrix],
    scaled_dense: np.ndarray,
) -> sparse.csr_matrix:
    blocks: list[sparse.spmatrix] = []
    for response_a, response_b in zip(response_a_matrices, response_b_matrices):
        difference = response_a - response_b
        blocks.extend([difference, abs(difference)])
    blocks.append(sparse.csr_matrix(scaled_dense, dtype=np.float32))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def swapped_compose_features(
    response_a_matrices: list[sparse.csr_matrix],
    response_b_matrices: list[sparse.csr_matrix],
    scaled_dense: np.ndarray,
) -> sparse.csr_matrix:
    blocks: list[sparse.spmatrix] = []
    for response_a, response_b in zip(response_a_matrices, response_b_matrices):
        difference = response_b - response_a
        blocks.extend([difference, abs(difference)])
    blocks.append(sparse.csr_matrix(scaled_dense, dtype=np.float32))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def fit_candidate(
    candidate: Candidate,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    epochs: int = 30,
    keep_objects: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_prompts, train_a, train_b = flattened_text_columns(training_frame)
    vectorizers = vectorizers_for(candidate)
    for vectorizer in vectorizers:
        print(f"[{candidate.name}] fitting {vectorizer.analyzer} TF-IDF", flush=True)
        vectorizer.fit(train_prompts + train_a + train_b)

    train_parts = [vectorize_parts(vectorizer, training_frame) for vectorizer in vectorizers]
    train_prompt_matrices = [parts[0] for parts in train_parts]
    train_a_matrices = [parts[1] for parts in train_parts]
    train_b_matrices = [parts[2] for parts in train_parts]

    train_dense, dense_names = raw_dense_features(
        training_frame,
        train_prompt_matrices[0],
        train_a_matrices[0],
        train_b_matrices[0],
    )
    swapped_dense = swapped_dense_features(train_dense, dense_names)
    swapped_names = dense_names
    if dense_names != swapped_names:
        raise RuntimeError("Dense feature order changed after A/B swapping.")
    scaler = StandardScaler()
    scaler.fit(np.vstack([train_dense, swapped_dense]))
    train_features = compose_features(
        train_a_matrices,
        train_b_matrices,
        scaler.transform(train_dense),
    )
    swapped_train_features = swapped_compose_features(
        train_a_matrices,
        train_b_matrices,
        scaler.transform(swapped_dense),
    )
    feature_count = int(train_features.shape[1])
    matrix_nonzero = int(train_features.nnz)
    del train_parts, train_prompt_matrices, train_a_matrices, train_b_matrices
    del train_dense, swapped_dense, train_prompts, train_a, train_b
    gc.collect()

    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=candidate.alpha,
        max_iter=1,
        tol=None,
        shuffle=True,
        random_state=seed,
        average=True,
    )
    # Train on the original and swapped matrices one epoch at a time. This is
    # equivalent to the intended augmentation while avoiding a second large
    # sparse matrix created by vstack at peak memory usage.
    swapped_targets = swap_target_indices(y_train)
    for epoch in range(epochs):
        model.partial_fit(
            train_features,
            y_train,
            classes=np.array([0, 1, 2]) if epoch == 0 else None,
        )
        model.partial_fit(swapped_train_features, swapped_targets)
    del train_features, swapped_train_features, swapped_targets
    gc.collect()

    validation_parts = [
        vectorize_parts(vectorizer, validation_frame) for vectorizer in vectorizers
    ]
    validation_prompt_matrices = [parts[0] for parts in validation_parts]
    validation_a_matrices = [parts[1] for parts in validation_parts]
    validation_b_matrices = [parts[2] for parts in validation_parts]
    validation_dense, _ = raw_dense_features(
        validation_frame,
        validation_prompt_matrices[0],
        validation_a_matrices[0],
        validation_b_matrices[0],
    )
    swapped_validation_dense = swapped_dense_features(validation_dense, dense_names)
    validation_features = compose_features(
        validation_a_matrices,
        validation_b_matrices,
        scaler.transform(validation_dense),
    )
    swapped_validation_features = swapped_compose_features(
        validation_a_matrices,
        validation_b_matrices,
        scaler.transform(swapped_validation_dense),
    )
    del validation_parts, validation_prompt_matrices
    del validation_a_matrices, validation_b_matrices
    del validation_dense, swapped_validation_dense
    gc.collect()
    original = ordered_predict_proba(model, validation_features)
    swapped_back = swap_probability_columns(
        ordered_predict_proba(model, swapped_validation_features)
    )
    metrics = evaluate_experiment_probabilities(
        y_validation, original, swapped_back
    )
    metrics.update(
        {
            "feature_count": feature_count,
            "vocabulary_sizes": [int(len(vectorizer.vocabulary_)) for vectorizer in vectorizers],
            "matrix_nonzero": matrix_nonzero,
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    result = {
        "candidate": candidate,
        "metrics": metrics,
    }
    if keep_objects:
        result.update(
            {
                "model": model,
                "vectorizers": vectorizers,
                "scaler": scaler,
                "dense_feature_names": dense_names,
                "original_validation_probability": original,
                "swapped_validation_probability": swapped_back,
            }
        )
    return result


def candidate_stages() -> list[list[Candidate]]:
    base = dict(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=75000,
        min_df=3,
        alpha=0.0003,
    )
    stages: list[list[Candidate]] = [
        [
            Candidate("control_word_1_2", "analyzer_and_ngram", **base),
            Candidate("word_1_1", "analyzer_and_ngram", **{**base, "ngram_range": (1, 1)}),
            Candidate("word_1_3", "analyzer_and_ngram", **{**base, "ngram_range": (1, 3)}),
            Candidate("char_3_5", "analyzer_and_ngram", **{**base, "analyzer": "char", "char_ngram_range": (3, 5)}),
            Candidate("char_3_6", "analyzer_and_ngram", **{**base, "analyzer": "char", "char_ngram_range": (3, 6)}),
            Candidate("word_char", "analyzer_and_ngram", **{**base, "analyzer": "word_char"}),
        ]
    ]
    return stages


def next_stage(best: Candidate, stage: str) -> list[Candidate]:
    if stage == "max_features":
        values = [50000, 100000]
        return [
            Candidate(
                f"{best.name}_max{value}", stage, best.analyzer,
                best.ngram_range, value, best.min_df, best.alpha,
                best.char_ngram_range, best.char_max_features,
            )
            for value in values
        ]
    if stage == "min_df":
        return [
            Candidate(
                f"{best.name}_min{value}", stage, best.analyzer,
                best.ngram_range, best.max_features, value, best.alpha,
                best.char_ngram_range, best.char_max_features,
            )
            for value in [1, 5]
        ]
    if stage == "alpha":
        return [
            Candidate(
                f"{best.name}_alpha{value:g}", stage, best.analyzer,
                best.ngram_range, best.max_features, best.min_df, value,
                best.char_ngram_range, best.char_max_features,
            )
            for value in [0.0001, 0.001]
        ]
    raise ValueError(f"Unknown sweep stage: {stage}")


def row_for_result(result: dict[str, Any]) -> dict[str, Any]:
    candidate: Candidate = result["candidate"]
    row = candidate.as_dict()
    row.update(result["metrics"])
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/E202609140001.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument("--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--initial-candidate-limit", type=int, default=None)
    parser.add_argument("--skip-refinement", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    epochs = int(args.epochs if args.epochs is not None else config["training"].get("epochs", 30))
    if epochs < 1:
        parser.error("--epochs must be a positive integer")
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    train, test = load_competition_data(args.data_dir)
    folds = load_frozen_folds(
        train, args.folds, args.split_config,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )
    fold_values = folds["fold"].to_numpy()
    is_training = np.isin(fold_values, list(roles.training_folds))
    is_validation = fold_values == roles.validation_fold
    y = target_indices(train)
    training_frame = train.loc[is_training].reset_index(drop=True)
    validation_frame = train.loc[is_validation].reset_index(drop=True)
    y_train = y[is_training]
    y_validation = y[is_validation]

    with ExperimentRun(
        args.config,
        project_root=PROJECT_ROOT,
        project_config_path=PROJECT_ROOT / "configs/project.json",
    ) as run:
        comparison_rows: list[dict[str, Any]] = []
        best_result: dict[str, Any] | None = None
        initial_candidates = candidate_stages()[0]
        if args.initial_candidate_limit is not None:
            initial_candidates = initial_candidates[:args.initial_candidate_limit]
        for candidates in [initial_candidates]:
            for candidate in candidates:
                result = fit_candidate(
                    candidate, training_frame, validation_frame,
                    y_train, y_validation, int(config["seed"]), epochs,
                )
                comparison_rows.append(row_for_result(result))
                previous_best = best_result
                if best_result is None or result["metrics"]["log_loss"] < best_result["metrics"]["log_loss"]:
                    best_result = result
                print(json.dumps(row_for_result(result), ensure_ascii=False))
                if result is not best_result:
                    del result
                if previous_best is not None and previous_best is not best_result:
                    del previous_best
                gc.collect()

        assert best_result is not None
        best_candidate: Candidate = best_result["candidate"]
        refinement_stages = [] if args.skip_refinement else ["max_features", "min_df", "alpha"]
        for stage in refinement_stages:
            stage_results: list[dict[str, Any]] = []
            for candidate in next_stage(best_candidate, stage):
                result = fit_candidate(
                    candidate, training_frame, validation_frame,
                    y_train, y_validation, int(config["seed"]), epochs,
                )
                comparison_rows.append(row_for_result(result))
                stage_results.append(result)
                print(json.dumps(row_for_result(result), ensure_ascii=False))
            stage_best = min(stage_results, key=lambda item: item["metrics"]["log_loss"])
            if stage_best["metrics"]["log_loss"] < best_result["metrics"]["log_loss"]:
                best_result = stage_best
                best_candidate = stage_best["candidate"]
            for item in stage_results:
                if item is not best_result:
                    del item
            gc.collect()

        # Recompute the selected candidate once so it remains available after
        # all non-selected candidate objects are released.
        selected = fit_candidate(
            best_candidate, training_frame, validation_frame,
            y_train, y_validation, int(config["seed"]), epochs, keep_objects=True,
        )
        comparison = pd.DataFrame(comparison_rows)
        comparison = comparison.sort_values("log_loss", ascending=True)
        comparison.to_csv(run.result_dir / "sparse_sweep_comparison.csv", index=False)

        run.log_metrics(
            {
                key: float(selected["metrics"][key])
                for key in [
                    "log_loss", "accuracy", "macro_f1", "ece_15",
                    "brier_score", "swap_error_l1",
                ]
            }
        )

        artifact_dir = run.artifact_dir
        model_path = artifact_dir / "sparse_estimator.joblib"
        joblib.dump(
            {
                "model": selected["model"],
                "vectorizers": selected["vectorizers"],
                "scaler": selected["scaler"],
                "dense_feature_names": selected["dense_feature_names"],
                "target_columns": TARGET_COLUMNS,
                "candidate": selected["candidate"].as_dict(),
                "training_folds": list(roles.training_folds),
                "validation_fold": roles.validation_fold,
            },
            model_path,
            compress=3,
        )
        validation_probability = normalize_probabilities(
            0.5 * (
                selected["original_validation_probability"]
                + selected["swapped_validation_probability"]
            )
        )
        validation_output = pd.DataFrame(
            {
                "id": validation_frame["id"].to_numpy(),
                "target": y_validation,
                TARGET_COLUMNS[0]: validation_probability[:, 0],
                TARGET_COLUMNS[1]: validation_probability[:, 1],
                TARGET_COLUMNS[2]: validation_probability[:, 2],
            }
        )
        validation_path = artifact_dir / "fold7_predictions.csv"
        validation_output.to_csv(validation_path, index=False)
        test_path = artifact_dir / "test_schema_predictions.csv"
        # The visible local test is only a three-row schema smoke test. Full
        # inference is reconstructed from the serialized estimator below.
        test_prompts, test_a, test_b = flattened_text_columns(test)
        test_parts = []
        for vectorizer in selected["vectorizers"]:
            test_parts.append(vectorize_parts(vectorizer, test))
        test_prompt_matrices = [part[0] for part in test_parts]
        test_a_matrices = [part[1] for part in test_parts]
        test_b_matrices = [part[2] for part in test_parts]
        test_dense, _ = raw_dense_features(
            test, test_prompt_matrices[0], test_a_matrices[0], test_b_matrices[0]
        )
        swapped_test_dense, _ = raw_dense_features(
            swapped_frame(test), test_prompt_matrices[0], test_b_matrices[0], test_a_matrices[0]
        )
        test_features = compose_features(
            test_a_matrices, test_b_matrices,
            selected["scaler"].transform(test_dense),
        )
        swapped_test_features = swapped_compose_features(
            test_a_matrices, test_b_matrices,
            selected["scaler"].transform(swapped_test_dense),
        )
        test_probability = normalize_probabilities(
            0.5 * (
                ordered_predict_proba(selected["model"], test_features)
                + swap_probability_columns(
                    ordered_predict_proba(selected["model"], swapped_test_features)
                )
            )
        )
        write_submission(test["id"].to_numpy(), test_probability, test_path)
        manifest = {
            "run_id": run.run_id,
            "experiment_id": config["experiment_id"],
            "seed": config["seed"],
            "class_order": list(TARGET_COLUMNS),
            "best_candidate": selected["candidate"].as_dict(),
            "artifacts": {
                "sparse_estimator": {"path": model_path.name, "sha256": sha256_file(model_path)},
                "fold7_predictions": {"path": validation_path.name, "sha256": sha256_file(validation_path)},
                "test_schema_predictions": {"path": test_path.name, "sha256": sha256_file(test_path)},
            },
        }
        (artifact_dir / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Best candidate: {selected['candidate'].name}")
        print(json.dumps(selected["metrics"], indent=2))
        print(f"Run: {run.run_id}")


if __name__ == "__main__":
    main()
