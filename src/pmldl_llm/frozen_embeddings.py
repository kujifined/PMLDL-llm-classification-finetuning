"""Frozen sentence-embedding baseline for the LLM preference task.

The encoder is never fine-tuned.  Embeddings are cached under ``artifacts/``
using the data files, model revision, pooling settings, and text-format
version as the cache key.  This makes repeated notebook runs incremental: a
smoke run can populate part of the cache and the full run computes only rows
that are still missing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import validate_fold_roles
from .constants import (
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_METADATA_PATH,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
    TARGET_COLUMNS,
)
from .data import (
    file_sha256,
    load_checksum_manifest,
    load_competition_data,
    target_indices,
    verify_competition_data_dir,
    swap_probability_columns,
)
from .evaluation import normalize_probabilities
from .notebook import ExperimentOutput, NotebookExperimentSetup
from .split import load_frozen_folds
from .text import flattened_text_columns


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# A stable Hub revision.  The wizard can override this through model@revision.
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MODEL_MAX_SEQ_LENGTH = 384
CACHE_FORMAT_VERSION = 1
TEXT_FORMAT_VERSION = 1


def _json_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


class EmbeddingCache:
    """Incremental cache for concatenated prompt/A/B embeddings."""

    def __init__(
        self,
        *,
        root: Path,
        model_name: str,
        model_revision: str,
        train_sha256: str,
        test_sha256: str,
        max_seq_length: int,
        normalize: bool = True,
    ) -> None:
        key = _json_hash(
            {
                "format_version": CACHE_FORMAT_VERSION,
                "model_name": model_name,
                "model_revision": model_revision,
                "max_seq_length": max_seq_length,
                "normalize": normalize,
                "text_format_version": TEXT_FORMAT_VERSION,
                "train_sha256": train_sha256,
                "test_sha256": test_sha256,
            }
        )
        self.key = key
        self.directory = root / "artifacts" / "cache" / "embeddings" / key
        self.normalize = normalize
        self.metadata = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": key,
            "model_name": model_name,
            "model_revision": model_revision,
            "max_seq_length": max_seq_length,
            "normalize_embeddings": normalize,
            "text_format_version": TEXT_FORMAT_VERSION,
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
        }

    def _path(self, split: str) -> Path:
        if split not in {"train", "test"}:
            raise ValueError("Embedding cache split must be train or test.")
        return self.directory / f"{split}.npz"

    def _load(self, split: str) -> tuple[np.ndarray, np.ndarray] | None:
        path = self._path(split)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cached:
            ids = cached["ids"].astype(str)
            features = cached["features"].astype(np.float32)
        if features.ndim != 2 or len(ids) != len(features):
            raise RuntimeError(f"Corrupted embedding cache: {path}")
        return ids, features

    def _save(self, split: str, ids: np.ndarray, features: np.ndarray) -> None:
        _atomic_npz(self._path(split), ids=ids.astype(str), features=features.astype(np.float32))
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def get_or_compute(
        self,
        *,
        split: str,
        ids: Iterable[str],
        text_parts: tuple[list[str], list[str], list[str]],
        encoder: Any,
        batch_size: int,
        show_progress_bar: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested_ids = np.asarray(list(ids), dtype=str)
        if len(requested_ids) != len(text_parts[0]):
            raise ValueError("Embedding ids and text rows must have equal length.")
        if len(set(requested_ids.tolist())) != len(requested_ids):
            raise ValueError("Embedding ids must be unique.")

        cached = self._load(split)
        existing_ids = np.asarray([], dtype=str)
        existing_features = np.empty((0, 0), dtype=np.float32)
        if cached is not None:
            existing_ids, existing_features = cached
        positions = {value: index for index, value in enumerate(existing_ids.tolist())}
        missing_indices = [
            index for index, value in enumerate(requested_ids.tolist()) if value not in positions
        ]
        if missing_indices:
            missing_parts = tuple(
                [part[index] for index in missing_indices] for part in text_parts
            )
            encoded_parts = [
                np.asarray(
                    encoder.encode(
                        values,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                    ),
                    dtype=np.float32,
                )
                for values in missing_parts
            ]
            missing_features = np.concatenate(encoded_parts, axis=1)
            if existing_features.size:
                all_ids = np.concatenate([existing_ids, requested_ids[missing_indices]])
                all_features = np.vstack([existing_features, missing_features])
            else:
                all_ids = requested_ids[missing_indices]
                all_features = missing_features
            self._save(split, all_ids, all_features)
            existing_ids, existing_features = all_ids, all_features
            positions = {value: index for index, value in enumerate(existing_ids.tolist())}

        selected = np.asarray([positions[value] for value in requested_ids], dtype=np.int64)
        return existing_features[selected], {
            "cache_key": self.key,
            "split": split,
            "requested_rows": int(len(requested_ids)),
            "computed_rows": int(len(missing_indices)),
            "cached_rows": int(len(requested_ids) - len(missing_indices)),
            "cache_path": self._path(split).as_posix(),
        }


def _build_swapped_features(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] % 3 != 0:
        raise ValueError("Expected concatenated prompt/A/B embeddings.")
    width = features.shape[1] // 3
    prompt, answer_a, answer_b = np.split(features, [width, 2 * width], axis=1)
    return np.concatenate([prompt, answer_b, answer_a], axis=1)


def _probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes_value = getattr(model, "classes_", None)
    if classes_value is None and hasattr(model, "steps"):
        classes_value = model.steps[-1][1].classes_
    classes = np.asarray(classes_value, dtype=np.int64)
    ordered = np.zeros((len(features), 3), dtype=np.float64)
    ordered[:, classes] = raw
    return normalize_probabilities(ordered)


def _smoke_subset(indices: np.ndarray, y: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    per_class = max(1, maximum // 3)
    chosen: list[int] = []
    for label in (0, 1, 2):
        chosen.extend(indices[y[indices] == label][:per_class].tolist())
    if len(chosen) < maximum:
        remainder = [index for index in indices.tolist() if index not in set(chosen)]
        chosen.extend(remainder[: maximum - len(chosen)])
    return np.asarray(chosen[:maximum], dtype=np.int64)


def _fit_logistic_hpo(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    seed: int,
    *,
    smoke_test: bool,
) -> tuple[Any, dict[str, Any]]:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    if smoke_test:
        model = pipeline.set_params(
            classifier__C=1.0,
            classifier__class_weight=None,
        )
        model.fit(features, targets)
        return model, {"mode": "smoke", "params": {"C": 1.0, "class_weight": None}}

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        param_grid={
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, targets, groups=groups)
    return search.best_estimator_, {
        "mode": "inner_group_cv",
        "best_params": search.best_params_,
        "best_cv_log_loss": float(-search.best_score_),
        "tested_configs": int(len(search.cv_results_["params"])),
    }


def _fit_mlp_hpo(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    seed: int,
    *,
    smoke_test: bool,
) -> tuple[Any, dict[str, Any]]:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    max_iter=150,
                    early_stopping=False,
                    random_state=seed,
                    tol=1e-4,
                ),
            ),
        ]
    )
    if smoke_test:
        model = pipeline.set_params(
            classifier__hidden_layer_sizes=(256,),
            classifier__alpha=1e-4,
            classifier__learning_rate_init=1e-3,
            classifier__batch_size=128,
        )
        model.fit(features, targets)
        return model, {
            "mode": "smoke",
            "params": {
                "hidden_layer_sizes": [256],
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "batch_size": 128,
            },
        }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        param_grid={
            "classifier__hidden_layer_sizes": [(128,), (256,), (256, 64)],
            "classifier__alpha": [1e-4, 1e-3],
            "classifier__learning_rate_init": [3e-4, 1e-3],
            "classifier__batch_size": [128],
        },
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, targets, groups=groups)
    best_params = dict(search.best_params_)
    if "classifier__hidden_layer_sizes" in best_params:
        best_params["classifier__hidden_layer_sizes"] = list(
            best_params["classifier__hidden_layer_sizes"]
        )
    return search.best_estimator_, {
        "mode": "inner_group_cv",
        "best_params": best_params,
        "best_cv_log_loss": float(-search.best_score_),
        "tested_configs": int(len(search.cv_results_["params"])),
    }


def _fit_catboost_hpo(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    seed: int,
    *,
    smoke_test: bool,
    config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit CatBoost with a small group-safe inner validation search.

    The outer selection fold is never used for choosing CatBoost parameters.
    A single deterministic group split keeps the optional boosting experiment
    affordable while preserving prompt-group isolation.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost requires the optional dependency. Install it with "
            "python -m pip install -e '.[boosting]'."
        ) from exc

    options = dict(config or {})
    task_type = str(options.get("task_type", "CPU")).upper()
    common: dict[str, Any] = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": int(options.get("thread_count", -1)),
        "task_type": task_type,
    }
    if options.get("devices") is not None:
        common["devices"] = str(options["devices"])

    def make_model(params: dict[str, Any]) -> Any:
        return CatBoostClassifier(**common, **params)

    smoke_params = {
        "iterations": int(options.get("smoke_iterations", 50)),
        "depth": int(options.get("smoke_depth", 4)),
        "learning_rate": float(options.get("smoke_learning_rate", 0.1)),
        "l2_leaf_reg": float(options.get("smoke_l2_leaf_reg", 3.0)),
    }
    if smoke_test:
        model = make_model(smoke_params)
        model.fit(features, targets)
        return model, {"mode": "smoke", "params": smoke_params, "task_type": task_type}

    default_configs = [
        {"iterations": 400, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 600, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
        {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 30.0},
    ]
    raw_configs = options.get("hpo_configs", default_configs)
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("catboost.hpo_configs must be a non-empty list of objects.")
    search_configs = [dict(item) for item in raw_configs]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
    inner_train, inner_validation = next(splitter.split(features, targets, groups=groups))
    scores: list[float] = []
    for params in search_configs:
        model = make_model(params)
        model.fit(features[inner_train], targets[inner_train])
        probabilities = _probabilities(model, features[inner_validation])
        scores.append(
            float(
                log_loss(
                    targets[inner_validation],
                    probabilities,
                    labels=[0, 1, 2],
                )
            )
        )
    best_index = int(np.argmin(scores))
    best_params = search_configs[best_index]
    best_model = make_model(best_params)
    best_model.fit(features, targets)
    return best_model, {
        "mode": "inner_group_holdout",
        "best_params": best_params,
        "best_inner_log_loss": scores[best_index],
        "tested_configs": len(search_configs),
        "inner_validation_rows": int(len(inner_validation)),
        "task_type": task_type,
    }


def _write_predictions(
    path: Path,
    ids: np.ndarray,
    targets: np.ndarray,
    original: np.ndarray,
    swapped_back: np.ndarray,
) -> None:
    averaged = normalize_probabilities(0.5 * (original + swapped_back))
    frame = pd.DataFrame(
        {
            "id": ids.astype(str),
            "target": targets.astype(int),
            "winner_model_a": averaged[:, 0],
            "winner_model_b": averaged[:, 1],
            "winner_tie": averaged[:, 2],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_submission(path: Path, ids: np.ndarray, probabilities: np.ndarray) -> None:
    frame = pd.DataFrame(normalize_probabilities(probabilities), columns=TARGET_COLUMNS)
    frame.insert(0, "id", ids.astype(str))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_frozen_embeddings_experiment(
    run: Any,
    setup: NotebookExperimentSetup,
    project_root: str | Path,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split_config_path: str | Path = DEFAULT_SPLIT_CONFIG,
    folds_path: str | Path = DEFAULT_FOLD_PATH,
    fold_metadata_path: str | Path = DEFAULT_FOLD_METADATA_PATH,
    checksum_manifest_path: str | Path = DEFAULT_CHECKSUM_MANIFEST,
) -> ExperimentOutput:
    """Run B3 and return the repository's canonical notebook output."""
    root = Path(project_root).resolve()
    data_path = Path(data_dir)
    split_path = Path(split_config_path)
    fold_path = Path(folds_path)
    metadata_path = Path(fold_metadata_path)
    manifest_path = Path(checksum_manifest_path)
    split_config = json.loads(split_path.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    expected_hashes = load_checksum_manifest(manifest_path)
    verify_competition_data_dir(data_path, expected_hashes)
    train, test = load_competition_data(data_path)
    folds = load_frozen_folds(
        train,
        fold_path,
        split_path,
        n_splits=roles.n_splits,
        metadata_path=metadata_path,
        dataset_hashes=expected_hashes,
    )
    fold_values = folds["fold"].to_numpy(dtype=np.int16)
    groups = folds["prompt_group"].to_numpy(dtype=str)
    targets = target_indices(train)
    train_positions = np.flatnonzero(np.isin(fold_values, roles.training_folds))
    validation_positions = np.flatnonzero(fold_values == roles.validation_fold)
    if setup.smoke_test:
        train_positions = _smoke_subset(train_positions, targets, 768)
        validation_positions = _smoke_subset(validation_positions, targets, 256)
    if len(train_positions) == 0 or len(validation_positions) == 0:
        raise RuntimeError("B3 requires non-empty training and selection folds.")

    model_name = setup.model_name or MODEL_NAME
    model_revision = setup.model_revision or MODEL_REVISION
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "B3 requires sentence-transformers. Install it with "
            "python -m pip install sentence-transformers."
        ) from exc
    encoder = SentenceTransformer(model_name, revision=model_revision)
    encoder.max_seq_length = MODEL_MAX_SEQ_LENGTH

    train_sha = file_sha256(data_path / "train.csv")
    test_sha = file_sha256(data_path / "test.csv")
    cache = EmbeddingCache(
        root=root,
        model_name=model_name,
        model_revision=model_revision,
        train_sha256=train_sha,
        test_sha256=test_sha,
        max_seq_length=MODEL_MAX_SEQ_LENGTH,
    )

    train_frame = train.reset_index(drop=True)
    test_frame = test.reset_index(drop=True)
    train_parts = flattened_text_columns(train_frame)
    test_parts = flattened_text_columns(test_frame)
    train_ids = train_frame["id"].astype(str).to_numpy()
    test_ids = test_frame["id"].astype(str).to_numpy()
    fit_ids = train_ids[train_positions]
    validation_ids = train_ids[validation_positions]
    fit_parts = tuple(
        [part[index] for index in train_positions] for part in train_parts
    )
    validation_parts = tuple(
        [part[index] for index in validation_positions] for part in train_parts
    )
    fit_features, fit_cache_info = cache.get_or_compute(
        split="train",
        ids=fit_ids,
        text_parts=fit_parts,
        encoder=encoder,
        batch_size=32,
        show_progress_bar=not setup.smoke_test,
    )
    validation_features, validation_cache_info = cache.get_or_compute(
        split="train",
        ids=validation_ids,
        text_parts=validation_parts,
        encoder=encoder,
        batch_size=32,
        show_progress_bar=False,
    )

    fit_targets = targets[train_positions]
    validation_targets = targets[validation_positions]
    fit_groups = groups[train_positions]
    classifiers = setup.training.get("classifiers", ["logistic", "mlp"])
    if not isinstance(classifiers, list) or not classifiers:
        raise ValueError("training.classifiers must be a non-empty list.")
    supported_classifiers = {"logistic", "mlp", "catboost"}
    unknown_classifiers = sorted(set(classifiers).difference(supported_classifiers))
    if unknown_classifiers:
        raise ValueError(f"Unsupported frozen-embedding classifiers: {unknown_classifiers}")

    fitted_models: dict[str, Any] = {}
    fit_info: dict[str, dict[str, Any]] = {}
    if "logistic" in classifiers:
        fitted_models["logistic"], fit_info["logistic"] = _fit_logistic_hpo(
            fit_features,
            fit_targets,
            fit_groups,
            int(setup.seed),
            smoke_test=setup.smoke_test,
        )
    if "mlp" in classifiers:
        fitted_models["mlp"], fit_info["mlp"] = _fit_mlp_hpo(
            fit_features,
            fit_targets,
            fit_groups,
            int(setup.seed),
            smoke_test=setup.smoke_test,
        )
    if "catboost" in classifiers:
        catboost_config = setup.training.get("catboost", {})
        if not isinstance(catboost_config, dict):
            raise ValueError("training.catboost must be an object.")
        fitted_models["catboost"], fit_info["catboost"] = _fit_catboost_hpo(
            fit_features,
            fit_targets,
            fit_groups,
            int(setup.seed),
            smoke_test=setup.smoke_test,
            config=catboost_config,
        )

    validation_swapped = _build_swapped_features(validation_features)
    candidates: dict[str, tuple[Any, np.ndarray, np.ndarray, float]] = {}
    for name, model in fitted_models.items():
        original = _probabilities(model, validation_features)
        swapped_back = swap_probability_columns(_probabilities(model, validation_swapped))
        averaged = normalize_probabilities(0.5 * (original + swapped_back))
        candidates[name] = (
            model,
            original,
            swapped_back,
            float(log_loss(validation_targets, averaged, labels=[0, 1, 2])),
        )
    best_name = min(candidates, key=lambda name: candidates[name][3])
    best_model, original_probabilities, swapped_back_probabilities, best_loss = candidates[
        best_name
    ]

    test_features = None
    test_cache_info: dict[str, Any] | None = None
    if not setup.smoke_test:
        test_features, test_cache_info = cache.get_or_compute(
            split="test",
            ids=test_ids,
            text_parts=test_parts,
            encoder=encoder,
            batch_size=32,
            show_progress_bar=True,
        )
        test_original = _probabilities(best_model, test_features)
        test_swapped = swap_probability_columns(
            _probabilities(best_model, _build_swapped_features(test_features))
        )
        test_probabilities = normalize_probabilities(0.5 * (test_original + test_swapped))
    else:
        test_probabilities = None

    prediction_path = run.artifact_path("predictions/validation_predictions.csv")
    _write_predictions(
        prediction_path,
        validation_ids,
        validation_targets,
        original_probabilities,
        swapped_back_probabilities,
    )
    model_path = run.artifact_path("models/frozen_embeddings_classifier.joblib")
    joblib.dump(best_model, model_path)
    hpo_path = run.artifact_path("reports/hpo_results.json")
    hpo_path.write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "embedding_revision": model_revision,
                "embedding_max_seq_length": MODEL_MAX_SEQ_LENGTH,
                "selected_classifier": best_name,
                "validation_log_loss": best_loss,
                "classifiers": list(fitted_models),
                "fit_info": fit_info,
                "cache": {
                    "train_fit": fit_cache_info,
                    "train_validation": validation_cache_info,
                    "test": test_cache_info,
                    "cache_metadata": cache.metadata,
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts: dict[str, str | Path] = {
        "predictions/validation_predictions.csv": prediction_path,
        "models/frozen_embeddings_classifier.joblib": model_path,
        "reports/hpo_results.json": hpo_path,
    }
    if test_probabilities is not None:
        submission_path = run.artifact_path("predictions/submission.csv")
        _write_submission(submission_path, test_ids, test_probabilities)
        artifacts["predictions/submission.csv"] = submission_path

    run.log_metrics(
        {f"{name}_log_loss": values[3] for name, values in candidates.items()},
        namespace="candidates",
    )
    run.log_metrics(
        {
            "embedding_rows_fit": float(len(train_positions)),
            "embedding_rows_validation": float(len(validation_positions)),
            "cache_rows_computed_fit": float(fit_cache_info["computed_rows"]),
            "cache_rows_computed_validation": float(
                validation_cache_info["computed_rows"]
            ),
        },
        namespace="resources",
    )
    return ExperimentOutput.from_predictions(
        y_true=validation_targets,
        original_probabilities=original_probabilities,
        swapped_back_probabilities=swapped_back_probabilities,
        artifacts=artifacts,
    )
