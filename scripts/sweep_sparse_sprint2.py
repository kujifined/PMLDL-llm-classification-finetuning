#!/usr/bin/env python3
"""Sequential one-factor-at-a-time TF-IDF sweep derived from the full B2 run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.config import validate_fold_roles  # noqa: E402
from pmldl_llm.constants import (  # noqa: E402
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
    TARGET_COLUMNS,
)
from pmldl_llm.data import (  # noqa: E402
    load_checksum_manifest,
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    target_indices,
    verify_checksum_manifest,
    verify_competition_data_dir,
)
from pmldl_llm.evaluation import (  # noqa: E402
    evaluate_experiment_probabilities,
    normalize_probabilities,
)
from pmldl_llm.experiment import ExperimentRun  # noqa: E402
from pmldl_llm.split import load_frozen_folds  # noqa: E402
from pmldl_llm.submission import write_submission  # noqa: E402
from pmldl_llm.tfidf_baseline import (  # noqa: E402
    TfidfChannelConfig,
    WordCharacterFeatureEncoder,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    word_ngram_range: tuple[int, int]
    character_ngram_range: tuple[int, int]
    word_max_features: int
    character_max_features: int
    word_min_df: int
    character_min_df: int
    max_df: float
    alpha: float

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["word_ngram_range"] = list(self.word_ngram_range)
        value["character_ngram_range"] = list(self.character_ngram_range)
        return value


SWEEP_FIELD_BY_STAGE = {
    "word_ngram_range": "word_ngram_range",
    "character_ngram_range": "character_ngram_range",
    "word_max_features": "word_max_features",
    "character_max_features": "character_max_features",
    "word_min_df": "word_min_df",
    "character_min_df": "character_min_df",
    "alpha": "alpha",
}

CANDIDATE_METRIC_NAMES = (
    "log_loss",
    "accuracy",
    "macro_f1",
    "ece_15",
    "brier_score",
    "swap_error_l1",
    "word_vocabulary_size",
    "character_vocabulary_size",
    "feature_columns",
    "training_matrix_nonzero",
    "candidate_runtime_seconds",
)

INTEGER_CANDIDATE_METRICS = {
    "word_vocabulary_size",
    "character_vocabulary_size",
    "feature_columns",
    "training_matrix_nonzero",
}


def _ordered_probabilities(
    model: SGDClassifier,
    features: sparse.csr_matrix,
) -> np.ndarray:
    raw = model.predict_proba(features)
    ordered = np.zeros((features.shape[0], 3), dtype=np.float64)
    ordered[:, model.classes_.astype(int)] = raw
    return normalize_probabilities(ordered)


def _averaged_probabilities(original: np.ndarray, swapped_back: np.ndarray) -> np.ndarray:
    return normalize_probabilities(0.5 * (original + swapped_back))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def control_from_parent(parent_config: Mapping[str, Any], alpha: float) -> Candidate:
    training = parent_config["training"]
    word = training["word_tfidf"]
    character = training["character_tfidf"]
    if character.get("analyzer", "char_wb") != "char_wb":
        raise ValueError("The Sprint 2 control must preserve B2 analyzer='char_wb'.")
    if float(word["max_df"]) != float(character["max_df"]):
        raise ValueError("The B2 word and character max_df values must match.")
    return Candidate(
        name="b2_control",
        stage="control",
        word_ngram_range=tuple(int(value) for value in word["ngram_range"]),
        character_ngram_range=tuple(
            int(value) for value in character["ngram_range"]
        ),
        word_max_features=int(word["max_features"]),
        character_max_features=int(character["max_features"]),
        word_min_df=int(word["min_df"]),
        character_min_df=int(character["min_df"]),
        max_df=float(word["max_df"]),
        alpha=float(alpha),
    )


def _normalized_sweep_value(stage: str, value: Any) -> Any:
    if stage.endswith("ngram_range"):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or value[0] < 1
            or value[0] > value[1]
        ):
            raise ValueError(f"training.sweep.{stage} values must be ordered pairs.")
        return tuple(value)
    if stage == "alpha":
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("Alpha candidates must be positive.")
        return numeric
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"training.sweep.{stage} values must be positive integers.")
    return value


def candidates_for_stage(
    reference: Candidate,
    stage: str,
    sweep: Mapping[str, Any],
) -> list[Candidate]:
    if stage not in SWEEP_FIELD_BY_STAGE:
        raise ValueError(f"Unsupported sweep stage: {stage!r}.")
    raw_values = sweep.get(stage)
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"training.sweep.{stage} must be a non-empty list.")
    field = SWEEP_FIELD_BY_STAGE[stage]
    current = getattr(reference, field)
    candidates: list[Candidate] = []
    for raw_value in raw_values:
        value = _normalized_sweep_value(stage, raw_value)
        if value == current:
            continue
        if isinstance(value, tuple):
            label = "_".join(str(item) for item in value)
        elif isinstance(value, float):
            label = f"{value:g}"
        else:
            label = str(value)
        candidates.append(
            replace(
                reference,
                name=f"{stage}_{label}",
                stage=stage,
                **{field: value},
            )
        )
    if not candidates:
        raise ValueError(f"training.sweep.{stage} contains no alternative to the control.")
    return candidates


def changed_candidate_fields(reference: Candidate, candidate: Candidate) -> set[str]:
    ignored = {"name", "stage"}
    reference_values = asdict(reference)
    candidate_values = asdict(candidate)
    return {
        key
        for key, value in reference_values.items()
        if key not in ignored and value != candidate_values[key]
    }


def fit_candidate(
    candidate: Candidate,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    average: bool,
    keep_objects: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"[{candidate.name}] fitting B2 word + char_wb TF-IDF", flush=True)
    word_config = TfidfChannelConfig(
        analyzer="word",
        ngram_range=candidate.word_ngram_range,
        max_features=candidate.word_max_features,
        min_df=candidate.word_min_df,
        max_df=candidate.max_df,
    )
    character_config = TfidfChannelConfig(
        analyzer="char_wb",
        ngram_range=candidate.character_ngram_range,
        max_features=candidate.character_max_features,
        min_df=candidate.character_min_df,
        max_df=candidate.max_df,
    )
    encoder, original_training, swapped_training = WordCharacterFeatureEncoder.fit(
        training_frame,
        word_config,
        character_config,
    )
    feature_columns = int(original_training.shape[1])
    training_matrix_nonzero = int(original_training.nnz)
    augmented_features = sparse.vstack(
        [original_training, swapped_training],
        format="csr",
        dtype=np.float32,
    )
    del original_training, swapped_training
    gc.collect()
    augmented_targets = np.concatenate(
        [training_targets, swap_target_indices(training_targets)]
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=candidate.alpha,
        max_iter=max_iter,
        tol=None,
        shuffle=True,
        random_state=seed,
        average=average,
    )
    model.fit(augmented_features, augmented_targets)
    del augmented_features, augmented_targets
    gc.collect()

    validation_features, swapped_validation_features = encoder.transform(
        validation_frame
    )
    original = _ordered_probabilities(model, validation_features)
    swapped_back = swap_probability_columns(
        _ordered_probabilities(model, swapped_validation_features)
    )
    metrics = evaluate_experiment_probabilities(
        validation_targets,
        original,
        swapped_back,
    )
    metrics.update(
        {
            "word_vocabulary_size": int(len(encoder.word_vectorizer.vocabulary_)),
            "character_vocabulary_size": int(
                len(encoder.character_vectorizer.vocabulary_)
            ),
            "feature_columns": feature_columns,
            "training_matrix_nonzero": training_matrix_nonzero,
            "candidate_runtime_seconds": float(time.perf_counter() - started),
        }
    )
    result: dict[str, Any] = {"candidate": candidate, "metrics": metrics}
    if keep_objects:
        result.update(
            {
                "encoder": encoder,
                "model": model,
                "original_validation_probabilities": original,
                "swapped_back_validation_probabilities": swapped_back,
            }
        )
    return result


def comparison_row(
    result: Mapping[str, Any],
    *,
    reference_name: str,
    reference_log_loss: float,
    parent_log_loss: float,
) -> dict[str, Any]:
    candidate: Candidate = result["candidate"]
    row = candidate.as_dict()
    row.update(result["metrics"])
    row.update(
        {
            "stage_reference": reference_name,
            "delta_log_loss_vs_stage_reference": float(
                result["metrics"]["log_loss"] - reference_log_loss
            ),
            "delta_log_loss_vs_b2": float(
                result["metrics"]["log_loss"] - parent_log_loss
            ),
            "stage_winner": "",
            "accepted_after_stage": False,
        }
    )
    return row


def write_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _comparison_pair(value: Any, *, field: str) -> tuple[int, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Resume row has invalid {field}: {value!r}.")
    return int(value[0]), int(value[1])


def candidate_from_comparison_row(row: Mapping[str, Any]) -> Candidate:
    return Candidate(
        name=str(row["name"]),
        stage=str(row["stage"]),
        word_ngram_range=_comparison_pair(
            row["word_ngram_range"], field="word_ngram_range"
        ),
        character_ngram_range=_comparison_pair(
            row["character_ngram_range"], field="character_ngram_range"
        ),
        word_max_features=int(row["word_max_features"]),
        character_max_features=int(row["character_max_features"]),
        word_min_df=int(row["word_min_df"]),
        character_min_df=int(row["character_min_df"]),
        max_df=float(row["max_df"]),
        alpha=float(row["alpha"]),
    )


def load_resume_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Resume comparison does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Resume comparison contains no completed candidates.")
    missing = {
        "name",
        "stage",
        "word_ngram_range",
        "character_ngram_range",
        "word_max_features",
        "character_max_features",
        "word_min_df",
        "character_min_df",
        "max_df",
        "alpha",
        *CANDIDATE_METRIC_NAMES,
    } - set(frame.columns)
    if missing:
        raise ValueError(
            f"Resume comparison is missing columns: {sorted(missing)}"
        )
    if frame["name"].duplicated().any():
        raise ValueError("Resume comparison contains duplicate candidate names.")

    results: dict[str, dict[str, Any]] = {}
    for raw_row in frame.to_dict(orient="records"):
        candidate = candidate_from_comparison_row(raw_row)
        metrics: dict[str, Any] = {}
        for name in CANDIDATE_METRIC_NAMES:
            value = float(raw_row[name])
            if not np.isfinite(value):
                raise ValueError(
                    f"Resume candidate {candidate.name!r} has non-finite {name}."
                )
            metrics[name] = int(value) if name in INTEGER_CANDIDATE_METRICS else value
        results[candidate.name] = {
            "candidate": candidate,
            "metrics": metrics,
        }
    return results


def resume_or_fit_candidate(
    candidate: Candidate,
    resume_results: Mapping[str, dict[str, Any]],
    reused_names: list[str],
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    average: bool,
) -> dict[str, Any]:
    resumed = resume_results.get(candidate.name)
    if resumed is None:
        return fit_candidate(
            candidate,
            training_frame,
            validation_frame,
            training_targets,
            validation_targets,
            seed=seed,
            max_iter=max_iter,
            average=average,
        )
    if resumed["candidate"] != candidate:
        raise ValueError(
            f"Resume candidate {candidate.name!r} does not match the current "
            "sequential sweep configuration."
        )
    reused_names.append(candidate.name)
    print(f"[{candidate.name}] reusing saved candidate metrics", flush=True)
    return resumed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/E202609140001.json",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument(
        "--checksum-manifest",
        type=Path,
        default=DEFAULT_CHECKSUM_MANIFEST,
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help=(
            "Read CSV files from an external read-only directory such as "
            "/kaggle/input. CSV hashes remain mandatory; only the local ZIP "
            "verification is skipped."
        ),
    )
    parser.add_argument(
        "--resume-comparison",
        type=Path,
        help=(
            "Reuse completed candidates from an interrupted "
            "sparse_sweep_comparison.csv. A new canonical run is created and "
            "all resumed rows are validated against the current sweep."
        ),
    )
    args = parser.parse_args()

    config = _load_json_object(args.config)
    training_config = config["training"]
    parent_config_path = PROJECT_ROOT / training_config["parent_config"]
    parent_config = _load_json_object(parent_config_path)
    if parent_config["experiment_id"] != config["parent_experiment_id"]:
        raise ValueError("Parent experiment ID and parent config disagree.")
    parent_run_id = str(training_config["parent_run_id"])
    parent_metrics_path = PROJECT_ROOT / "results/runs" / parent_run_id / "metrics.json"
    parent_metrics = _load_json_object(parent_metrics_path)
    parent_log_loss = float(parent_metrics["summary"]["validation"]["log_loss"])
    selected_alpha = float(
        parent_metrics["summary"]["diagnostics"]["selected_alpha"]
    )
    control = control_from_parent(parent_config, selected_alpha)

    split_config = _load_json_object(args.split_config)
    roles = validate_fold_roles(split_config)
    dataset_hashes = (
        load_checksum_manifest(args.checksum_manifest)
        if args.external_data
        else verify_checksum_manifest(args.checksum_manifest)
    )
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
    training_mask = np.isin(fold_values, list(roles.training_folds))
    validation_mask = fold_values == roles.validation_fold
    targets = target_indices(train)
    training_frame = train.loc[training_mask].reset_index(drop=True)
    validation_frame = train.loc[validation_mask].reset_index(drop=True)
    training_targets = targets[training_mask]
    validation_targets = targets[validation_mask]

    classifier_config = training_config["classifier"]
    max_iter = int(classifier_config["max_iter"])
    average = bool(classifier_config["average"])
    stage_order = list(training_config["stage_order"])
    sweep = training_config["sweep"]
    resume_comparison = (
        args.resume_comparison.resolve() if args.resume_comparison else None
    )
    resume_results = (
        load_resume_results(resume_comparison) if resume_comparison else {}
    )
    reused_names: list[str] = []

    with ExperimentRun(
        args.config,
        project_root=PROJECT_ROOT,
        project_config_path=PROJECT_ROOT / "configs/project.json",
    ) as run:
        comparison_path = run.result_dir / "sparse_sweep_comparison.csv"
        rows: list[dict[str, Any]] = []
        best_result = resume_or_fit_candidate(
            control,
            resume_results,
            reused_names,
            training_frame,
            validation_frame,
            training_targets,
            validation_targets,
            seed=int(config["seed"]),
            max_iter=max_iter,
            average=average,
        )
        control_row = comparison_row(
            best_result,
            reference_name=parent_run_id,
            reference_log_loss=parent_log_loss,
            parent_log_loss=parent_log_loss,
        )
        control_row["stage_winner"] = control.name
        control_row["accepted_after_stage"] = True
        rows.append(control_row)
        write_comparison(rows, comparison_path)
        print(json.dumps(control_row, ensure_ascii=False), flush=True)

        for stage in stage_order:
            reference_result = best_result
            reference: Candidate = reference_result["candidate"]
            reference_loss = float(reference_result["metrics"]["log_loss"])
            stage_results: list[dict[str, Any]] = []
            stage_start = len(rows)
            for candidate in candidates_for_stage(reference, stage, sweep):
                changed = changed_candidate_fields(reference, candidate)
                if changed != {SWEEP_FIELD_BY_STAGE[stage]}:
                    raise RuntimeError(
                        f"{candidate.name} changes {sorted(changed)}, expected only {stage}."
                    )
                result = resume_or_fit_candidate(
                    candidate,
                    resume_results,
                    reused_names,
                    training_frame,
                    validation_frame,
                    training_targets,
                    validation_targets,
                    seed=int(config["seed"]),
                    max_iter=max_iter,
                    average=average,
                )
                stage_results.append(result)
                row = comparison_row(
                    result,
                    reference_name=reference.name,
                    reference_log_loss=reference_loss,
                    parent_log_loss=parent_log_loss,
                )
                rows.append(row)
                write_comparison(rows, comparison_path)
                print(json.dumps(row, ensure_ascii=False), flush=True)

            stage_winner = min(
                [reference_result, *stage_results],
                key=lambda item: item["metrics"]["log_loss"],
            )
            winner_name = stage_winner["candidate"].name
            for row in rows[stage_start:]:
                row["stage_winner"] = winner_name
                row["accepted_after_stage"] = row["name"] == winner_name
            best_result = stage_winner
            write_comparison(rows, comparison_path)
            del stage_results
            gc.collect()

        unused_resume_candidates = set(resume_results) - set(reused_names)
        if unused_resume_candidates:
            raise ValueError(
                "Resume comparison contains candidates that do not belong to "
                f"the reconstructed sweep: {sorted(unused_resume_candidates)}"
            )

        best_candidate: Candidate = best_result["candidate"]
        selected = fit_candidate(
            best_candidate,
            training_frame,
            validation_frame,
            training_targets,
            validation_targets,
            seed=int(config["seed"]),
            max_iter=max_iter,
            average=average,
            keep_objects=True,
        )
        best_config_path = run.result_dir / "best_tfidf_config.json"
        best_config_path.write_text(
            json.dumps(
                {
                    "parent_experiment_id": config["parent_experiment_id"],
                    "parent_run_id": parent_run_id,
                    "parent_log_loss": parent_log_loss,
                    "selected_candidate": best_candidate.as_dict(),
                    "selected_metrics": selected["metrics"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        run.log_metrics(
            {
                key: float(selected["metrics"][key])
                for key in (
                    "log_loss",
                    "accuracy",
                    "macro_f1",
                    "ece_15",
                    "brier_score",
                    "swap_error_l1",
                )
            }
        )
        run.log_metrics(
            {
                "training_rows": len(training_frame),
                "validation_rows": len(validation_frame),
                "word_vocabulary_size": selected["metrics"]["word_vocabulary_size"],
                "character_vocabulary_size": selected["metrics"][
                    "character_vocabulary_size"
                ],
                "feature_columns": selected["metrics"]["feature_columns"],
                "training_matrix_nonzero": selected["metrics"][
                    "training_matrix_nonzero"
                ],
                "selected_alpha": best_candidate.alpha,
                "resumed_candidates": len(reused_names),
                "delta_log_loss_vs_b2": selected["metrics"]["log_loss"]
                - parent_log_loss,
            },
            namespace="diagnostics",
        )

        model_path = run.artifact_path("sparse_estimator.joblib")
        joblib.dump(
            {
                "model": selected["model"],
                "encoder": selected["encoder"],
                "target_columns": TARGET_COLUMNS,
                "candidate": best_candidate.as_dict(),
                "parent_experiment_id": config["parent_experiment_id"],
                "parent_run_id": parent_run_id,
                "training_folds": list(roles.training_folds),
                "validation_fold": roles.validation_fold,
            },
            model_path,
            compress=3,
        )
        validation_probability = _averaged_probabilities(
            selected["original_validation_probabilities"],
            selected["swapped_back_validation_probabilities"],
        )
        validation_path = run.artifact_path("fold7_predictions.csv")
        pd.DataFrame(
            {
                "id": validation_frame["id"].to_numpy(),
                "target": validation_targets,
                TARGET_COLUMNS[0]: validation_probability[:, 0],
                TARGET_COLUMNS[1]: validation_probability[:, 1],
                TARGET_COLUMNS[2]: validation_probability[:, 2],
            }
        ).to_csv(validation_path, index=False)

        test_features, swapped_test_features = selected["encoder"].transform(test)
        original_test = _ordered_probabilities(selected["model"], test_features)
        swapped_back_test = swap_probability_columns(
            _ordered_probabilities(selected["model"], swapped_test_features)
        )
        test_probability = _averaged_probabilities(original_test, swapped_back_test)
        test_path = run.artifact_path("test_schema_predictions.csv")
        write_submission(test["id"].to_numpy(), test_probability, test_path)

        manifest = {
            "run_id": run.run_id,
            "experiment_id": config["experiment_id"],
            "parent_experiment_id": config["parent_experiment_id"],
            "parent_run_id": parent_run_id,
            "seed": config["seed"],
            "class_order": list(TARGET_COLUMNS),
            "best_candidate": best_candidate.as_dict(),
            "artifacts": {
                "sparse_estimator": {
                    "path": model_path.name,
                    "sha256": _sha256_file(model_path),
                },
                "fold7_predictions": {
                    "path": validation_path.name,
                    "sha256": _sha256_file(validation_path),
                },
                "test_schema_predictions": {
                    "path": test_path.name,
                    "sha256": _sha256_file(test_path),
                },
            },
        }
        manifest_path = run.artifact_path("model_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts_to_log = [
            ("sparse_sweep_comparison.csv", comparison_path),
            ("best_tfidf_config.json", best_config_path),
            (model_path.name, model_path),
            (validation_path.name, validation_path),
            (test_path.name, test_path),
            (manifest_path.name, manifest_path),
        ]
        if resume_comparison is not None:
            source_run_path = resume_comparison.parent / "run.json"
            source_run = (
                _load_json_object(source_run_path)
                if source_run_path.is_file()
                else None
            )
            comparison_label = resume_comparison.name
            if source_run is not None and source_run.get("result_directory"):
                comparison_label = (
                    f"{source_run['result_directory']}/{resume_comparison.name}"
                )
            resume_source_path = run.result_dir / "resume_source.json"
            resume_source_path.write_text(
                json.dumps(
                    {
                        "comparison_path": comparison_label,
                        "comparison_sha256": _sha256_file(resume_comparison),
                        "source_run": source_run,
                        "reused_candidates": reused_names,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts_to_log.append((resume_source_path.name, resume_source_path))
        for artifact_name, artifact_path in artifacts_to_log:
            run.log_artifact(artifact_name, artifact_path)
        print(f"Best candidate: {best_candidate.name}")
        print(json.dumps(selected["metrics"], indent=2, ensure_ascii=False))
        print(f"Comparison: {comparison_path}")
        print(f"Best config: {best_config_path}")
        print(f"Run: {run.run_id}")


if __name__ == "__main__":
    main()
