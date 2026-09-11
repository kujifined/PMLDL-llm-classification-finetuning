from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .config import validate_fold_roles
from .constants import TARGET_COLUMNS
from .data import (
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    swapped_frame,
    target_indices,
    verify_competition_data_dir,
    verify_checksum_manifest,
)
from .evaluation import evaluate_experiment_probabilities, normalize_probabilities
from .features import build_bias_features
from .notebook import ExperimentOutput, NotebookExperimentSetup
from .split import load_frozen_folds
from .text import flattened_text_columns


@dataclass(frozen=True)
class TfidfChannelConfig:
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    max_df: float


@dataclass(frozen=True)
class LinearClassifierConfig:
    loss: str
    alpha_candidates: tuple[float, ...]
    max_iter: int
    average: bool


@dataclass
class WordCharacterFeatureEncoder:
    word_vectorizer: TfidfVectorizer
    character_vectorizer: TfidfVectorizer
    dense_scaler: StandardScaler
    dense_feature_names: list[str]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        word_config: TfidfChannelConfig,
        character_config: TfidfChannelConfig,
    ) -> tuple["WordCharacterFeatureEncoder", sparse.csr_matrix, sparse.csr_matrix]:
        prompt, response_a, response_b = flattened_text_columns(frame)
        corpus = prompt + response_a + response_b
        word_vectorizer = _build_vectorizer(word_config)
        character_vectorizer = _build_vectorizer(character_config)
        word_vectorizer.fit(corpus)
        character_vectorizer.fit(corpus)

        word_parts = _vectorize_parts(
            word_vectorizer,
            prompt,
            response_a,
            response_b,
        )
        character_parts = _vectorize_parts(
            character_vectorizer,
            prompt,
            response_a,
            response_b,
        )
        original_dense, dense_feature_names = _dense_features(
            frame,
            word_parts,
            character_parts,
        )
        swapped_dense, swapped_names = _dense_features(
            swapped_frame(frame),
            _swap_response_parts(word_parts),
            _swap_response_parts(character_parts),
        )
        if swapped_names != dense_feature_names:
            raise RuntimeError("Dense feature order changed after swapping A and B.")

        dense_scaler = StandardScaler()
        dense_scaler.fit(np.vstack([original_dense, swapped_dense]))
        encoder = cls(
            word_vectorizer=word_vectorizer,
            character_vectorizer=character_vectorizer,
            dense_scaler=dense_scaler,
            dense_feature_names=dense_feature_names,
        )
        original_features = _compose_features(
            word_parts,
            character_parts,
            dense_scaler.transform(original_dense),
        )
        swapped_features = _compose_features(
            _swap_response_parts(word_parts),
            _swap_response_parts(character_parts),
            dense_scaler.transform(swapped_dense),
        )
        return encoder, original_features, swapped_features

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        prompt, response_a, response_b = flattened_text_columns(frame)
        word_parts = _vectorize_parts(
            self.word_vectorizer,
            prompt,
            response_a,
            response_b,
        )
        character_parts = _vectorize_parts(
            self.character_vectorizer,
            prompt,
            response_a,
            response_b,
        )
        original_dense, names = _dense_features(
            frame,
            word_parts,
            character_parts,
        )
        swapped_dense, swapped_names = _dense_features(
            swapped_frame(frame),
            _swap_response_parts(word_parts),
            _swap_response_parts(character_parts),
        )
        if names != self.dense_feature_names or swapped_names != names:
            raise RuntimeError("Dense feature schema differs from the fitted encoder.")
        original_features = _compose_features(
            word_parts,
            character_parts,
            self.dense_scaler.transform(original_dense),
        )
        swapped_features = _compose_features(
            _swap_response_parts(word_parts),
            _swap_response_parts(character_parts),
            self.dense_scaler.transform(swapped_dense),
        )
        return original_features, swapped_features


def run_tfidf_experiment(
    run: Any,
    setup: NotebookExperimentSetup,
    project_root: str | Path,
) -> ExperimentOutput:
    root = Path(project_root).resolve()
    data_dir = root / "data" / "llm-classification-finetuning"
    checksum_manifest = root / "data" / "checksums.sha256"
    split_config_path = root / "configs" / "split.json"
    fold_path = root / "data" / "splits" / "folds.csv"
    word_config = _channel_config(setup.training, "word_tfidf", "word")
    character_config = _channel_config(
        setup.training,
        "character_tfidf",
        "char_wb",
    )
    classifier_config = _classifier_config(setup.training)

    split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(checksum_manifest)
    verify_competition_data_dir(data_dir, dataset_hashes)
    train, _ = load_competition_data(data_dir)
    folds = load_frozen_folds(
        train,
        fold_path,
        split_config_path,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )
    fold_values = folds["fold"].to_numpy()
    training_mask = np.isin(fold_values, roles.training_folds)
    validation_mask = fold_values == roles.validation_fold
    all_targets = target_indices(train)
    training_frame = train.loc[training_mask].reset_index(drop=True)
    validation_frame = train.loc[validation_mask].reset_index(drop=True)
    training_targets = all_targets[training_mask]
    validation_targets = all_targets[validation_mask]

    if setup.smoke_test:
        training_frame, training_targets = _stratified_sample(
            training_frame,
            training_targets,
            int(setup.training["smoke_train_rows"]),
            setup.seed,
        )
        validation_frame, validation_targets = _stratified_sample(
            validation_frame,
            validation_targets,
            int(setup.training["smoke_validation_rows"]),
            setup.seed + 1,
        )

    encoder, training_features, swapped_training_features = (
        WordCharacterFeatureEncoder.fit(
            training_frame,
            word_config,
            character_config,
        )
    )
    feature_columns = int(training_features.shape[1])
    training_matrix_nonzero = int(training_features.nnz)
    augmented_features = sparse.vstack(
        [training_features, swapped_training_features],
        format="csr",
        dtype=np.float32,
    )
    del training_features, swapped_training_features
    augmented_targets = np.concatenate(
        [training_targets, swap_target_indices(training_targets)]
    )
    validation_features, swapped_validation_features = encoder.transform(
        validation_frame
    )
    (
        model,
        selected_alpha,
        candidate_metrics,
        original_probabilities,
        swapped_back_probabilities,
    ) = _fit_select_model(
        augmented_features=augmented_features,
        augmented_targets=augmented_targets,
        validation_features=validation_features,
        swapped_validation_features=swapped_validation_features,
        validation_targets=validation_targets,
        config=classifier_config,
        seed=setup.seed,
    )
    averaged_probabilities = _average_order_probabilities(
        original_probabilities,
        swapped_back_probabilities,
    )

    model_path = run.artifact_path("models/word_character_tfidf.joblib")
    predictions_path = run.artifact_path("predictions/validation.csv")
    metadata_path = run.artifact_path("feature_metadata.json")
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "target_columns": TARGET_COLUMNS,
            "experiment_config": setup.training,
        },
        model_path,
    )
    pd.DataFrame(
        {
            "id": validation_frame["id"].to_numpy(),
            "target": validation_targets,
            TARGET_COLUMNS[0]: averaged_probabilities[:, 0],
            TARGET_COLUMNS[1]: averaged_probabilities[:, 1],
            TARGET_COLUMNS[2]: averaged_probabilities[:, 2],
        }
    ).to_csv(predictions_path, index=False)
    metadata = {
        "smoke_test": setup.smoke_test,
        "training_rows": int(len(training_frame)),
        "validation_rows": int(len(validation_frame)),
        "word_vocabulary_size": int(len(encoder.word_vectorizer.vocabulary_)),
        "character_vocabulary_size": int(
            len(encoder.character_vectorizer.vocabulary_)
        ),
        "feature_columns": feature_columns,
        "training_matrix_nonzero": training_matrix_nonzero,
        "word_tfidf": setup.training["word_tfidf"],
        "character_tfidf": setup.training["character_tfidf"],
        "classifier": setup.training["classifier"],
        "selected_alpha": selected_alpha,
        "candidate_metrics": candidate_metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run.log_metrics(
        {
            "training_rows": metadata["training_rows"],
            "validation_rows": metadata["validation_rows"],
            "word_vocabulary_size": metadata["word_vocabulary_size"],
            "character_vocabulary_size": metadata[
                "character_vocabulary_size"
            ],
            "feature_columns": metadata["feature_columns"],
            "training_matrix_nonzero": metadata["training_matrix_nonzero"],
            "selected_alpha": metadata["selected_alpha"],
        },
        namespace="diagnostics",
    )
    return ExperimentOutput.from_predictions(
        y_true=validation_targets,
        original_probabilities=original_probabilities,
        swapped_back_probabilities=swapped_back_probabilities,
        artifacts={
            "models/word_character_tfidf.joblib": model_path,
            "predictions/validation.csv": predictions_path,
            "feature_metadata.json": metadata_path,
        },
    )


def _build_vectorizer(config: TfidfChannelConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=config.analyzer,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def _vectorize_parts(
    vectorizer: TfidfVectorizer,
    prompt: list[str],
    response_a: list[str],
    response_b: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    return (
        vectorizer.transform(prompt).tocsr(),
        vectorizer.transform(response_a).tocsr(),
        vectorizer.transform(response_b).tocsr(),
    )


def _swap_response_parts(
    parts: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    prompt, response_a, response_b = parts
    return prompt, response_b, response_a


def _dense_features(
    frame: pd.DataFrame,
    word_parts: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
    character_parts: tuple[
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
    ],
) -> tuple[np.ndarray, list[str]]:
    bias_features, bias_names = build_bias_features(frame)
    columns = [bias_features]
    names = list(bias_names)
    for prefix, parts in (("word", word_parts), ("character", character_parts)):
        prompt, response_a, response_b = parts
        similarity_a = np.asarray(prompt.multiply(response_a).sum(axis=1)).ravel()
        similarity_b = np.asarray(prompt.multiply(response_b).sum(axis=1)).ravel()
        columns.append(
            np.column_stack(
                [
                    similarity_a,
                    similarity_b,
                    similarity_a - similarity_b,
                    np.abs(similarity_a - similarity_b),
                ]
            )
        )
        names.extend(
            [
                f"{prefix}_prompt_response_a_cosine",
                f"{prefix}_prompt_response_b_cosine",
                f"{prefix}_prompt_similarity_difference",
                f"{prefix}_absolute_prompt_similarity_difference",
            ]
        )
    dense = np.column_stack(columns)
    if not np.isfinite(dense).all():
        raise ValueError("Dense TF-IDF features contain non-finite values.")
    return dense, names


def _compose_features(
    word_parts: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
    character_parts: tuple[
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
    ],
    scaled_dense: np.ndarray,
) -> sparse.csr_matrix:
    _, word_a, word_b = word_parts
    _, character_a, character_b = character_parts
    word_difference = word_a - word_b
    character_difference = character_a - character_b
    return sparse.hstack(
        [
            word_difference,
            abs(word_difference),
            character_difference,
            abs(character_difference),
            sparse.csr_matrix(scaled_dense, dtype=np.float32),
        ],
        format="csr",
        dtype=np.float32,
    )


def _ordered_probabilities(
    model: SGDClassifier,
    features: sparse.csr_matrix,
) -> np.ndarray:
    raw = model.predict_proba(features)
    ordered = np.zeros((features.shape[0], 3), dtype=np.float64)
    ordered[:, model.classes_.astype(int)] = raw
    return normalize_probabilities(ordered)


def _average_order_probabilities(
    original: np.ndarray,
    swapped_back: np.ndarray,
) -> np.ndarray:
    return normalize_probabilities(0.5 * (original + swapped_back))


def _fit_select_model(
    *,
    augmented_features: sparse.csr_matrix,
    augmented_targets: np.ndarray,
    validation_features: sparse.csr_matrix,
    swapped_validation_features: sparse.csr_matrix,
    validation_targets: np.ndarray,
    config: LinearClassifierConfig,
    seed: int,
) -> tuple[
    SGDClassifier,
    float,
    dict[str, dict[str, float]],
    np.ndarray,
    np.ndarray,
]:
    candidates: list[
        tuple[
            SGDClassifier,
            float,
            dict[str, float],
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for alpha in config.alpha_candidates:
        model = SGDClassifier(
            loss=config.loss,
            penalty="l2",
            alpha=alpha,
            max_iter=config.max_iter,
            tol=None,
            shuffle=True,
            random_state=seed,
            average=config.average,
        )
        model.fit(augmented_features, augmented_targets)
        original = _ordered_probabilities(model, validation_features)
        swapped_back = swap_probability_columns(
            _ordered_probabilities(model, swapped_validation_features)
        )
        metrics = evaluate_experiment_probabilities(
            validation_targets,
            original,
            swapped_back,
        )
        candidates.append((model, alpha, metrics, original, swapped_back))

    best = min(candidates, key=lambda candidate: candidate[2]["log_loss"])
    model, selected_alpha, _, original, swapped_back = best
    candidate_metrics = {
        f"alpha={alpha:g}": metrics
        for _, alpha, metrics, _, _ in candidates
    }
    return model, selected_alpha, candidate_metrics, original, swapped_back


def _channel_config(
    training: Mapping[str, Any],
    key: str,
    default_analyzer: str,
) -> TfidfChannelConfig:
    raw = training.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"training.{key} must be an object.")
    ngram_range = raw.get("ngram_range")
    if (
        not isinstance(ngram_range, list)
        or len(ngram_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in ngram_range
        )
        or ngram_range[0] < 1
        or ngram_range[0] > ngram_range[1]
    ):
        raise ValueError(
            f"training.{key}.ngram_range must contain two ordered integers."
        )
    analyzer = raw.get("analyzer", default_analyzer)
    if analyzer not in {"word", "char", "char_wb"}:
        raise ValueError(f"Unsupported analyzer for training.{key}: {analyzer!r}.")
    max_features = _positive_int(
        raw.get("max_features"),
        f"training.{key}.max_features",
    )
    min_df = _positive_int(raw.get("min_df"), f"training.{key}.min_df")
    max_df = raw.get("max_df")
    if isinstance(max_df, bool) or not isinstance(max_df, (int, float)):
        raise ValueError(f"training.{key}.max_df must be numeric.")
    max_df = float(max_df)
    if not 0.0 < max_df <= 1.0:
        raise ValueError(f"training.{key}.max_df must be in (0, 1].")
    return TfidfChannelConfig(
        analyzer=str(analyzer),
        ngram_range=(ngram_range[0], ngram_range[1]),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )


def _classifier_config(training: Mapping[str, Any]) -> LinearClassifierConfig:
    raw = training.get("classifier")
    if not isinstance(raw, Mapping):
        raise ValueError("training.classifier must be an object.")
    loss = raw.get("loss")
    if loss != "log_loss":
        raise ValueError("The B2 classifier loss must be 'log_loss'.")
    alpha_candidates = raw.get("alpha_candidates")
    if not isinstance(alpha_candidates, list) or not alpha_candidates:
        raise ValueError(
            "training.classifier.alpha_candidates must be a non-empty list."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        for value in alpha_candidates
    ):
        raise ValueError(
            "training.classifier.alpha_candidates must contain positive numbers."
        )
    average = raw.get("average")
    if not isinstance(average, bool):
        raise ValueError("training.classifier.average must be boolean.")
    return LinearClassifierConfig(
        loss=loss,
        alpha_candidates=tuple(float(value) for value in alpha_candidates),
        max_iter=_positive_int(
            raw.get("max_iter"),
            "training.classifier.max_iter",
        ),
        average=average,
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _stratified_sample(
    frame: pd.DataFrame,
    targets: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if limit >= len(frame):
        return frame.reset_index(drop=True), np.asarray(targets)
    if limit < 3:
        raise ValueError("Smoke sample must contain at least three rows.")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    base = limit // 3
    for label in range(3):
        candidates = np.flatnonzero(targets == label)
        if len(candidates) == 0:
            raise ValueError(f"Smoke source contains no examples of class {label}.")
        take = min(base, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remaining = limit - len(selected)
    if remaining:
        pool = np.setdiff1d(np.arange(len(frame)), np.asarray(selected))
        selected.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    indices = np.sort(np.asarray(selected, dtype=np.int64))
    return frame.iloc[indices].reset_index(drop=True), np.asarray(targets)[indices]
