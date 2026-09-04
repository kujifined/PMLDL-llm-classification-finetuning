#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    file_sha256,
    load_competition_data,
    target_indices,
    verify_checksum_manifest,
    verify_competition_data_dir,
)
from pmldl_llm.evaluation import evaluate_probabilities, normalize_probabilities
from pmldl_llm.provenance import build_provenance, path_label
from pmldl_llm.split import load_frozen_folds


def aligned_predictions(
    left_path: Path,
    right_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    left = pd.read_csv(left_path).sort_values("id").reset_index(drop=True)
    right = pd.read_csv(right_path).sort_values("id").reset_index(drop=True)
    if not left["id"].is_unique or not right["id"].is_unique:
        raise ValueError("Prediction files must contain unique ids.")
    if not left["id"].equals(right["id"]):
        raise ValueError("Prediction files contain different validation ids.")
    if not left["target"].equals(right["target"]):
        raise ValueError("Prediction files contain different validation targets.")
    columns = list(TARGET_COLUMNS)
    return (
        left[["id", "target"]],
        left["target"].to_numpy(dtype=np.int64),
        left[columns].to_numpy(dtype=np.float64),
        right[columns].to_numpy(dtype=np.float64),
    )


def row_log_loss(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    return -np.log(probabilities[np.arange(len(y)), y])


def grouped_bootstrap_delta(
    y: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, float]:
    delta = row_log_loss(y, candidate_probability) - row_log_loss(
        y, reference_probability
    )
    summary = pd.DataFrame({"group": groups, "delta": delta}).groupby("group")[
        "delta"
    ].agg(["sum", "count"])
    group_sum = summary["sum"].to_numpy()
    group_count = summary["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(summary), size=len(summary))
        draws[index] = group_sum[sampled].sum() / group_count[sampled].sum()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate_minus_reference": float(delta.mean()),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "n_prompt_groups": int(len(summary)),
        "n_resamples": int(n_resamples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structural",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "bias_baseline"
            / "validation_predictions.csv"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--sparse",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "sparse_baseline"
            / "validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=DEFAULT_FOLD_PATH,
    )
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_blend",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "baselines"
            / "baseline_blend"
            / "metrics.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    train, _ = load_competition_data(args.data_dir)
    folds = load_frozen_folds(
        train,
        args.folds,
        args.split_config,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )

    identity, y, structural, sparse_probability = aligned_predictions(
        args.structural, args.sparse
    )
    candidates = []
    best_loss = float("inf")
    best_weight = None
    best_probability = None
    for structural_weight in np.linspace(0.0, 1.0, 101):
        probability = normalize_probabilities(
            structural_weight * structural
            + (1.0 - structural_weight) * sparse_probability
        )
        evaluation = evaluate_probabilities(y, probability)
        candidates.append(
            {
                "structural_weight": float(structural_weight),
                "sparse_weight": float(1.0 - structural_weight),
                "log_loss": evaluation["log_loss"],
            }
        )
        if evaluation["log_loss"] < best_loss:
            best_loss = float(evaluation["log_loss"])
            best_weight = float(structural_weight)
            best_probability = probability

    if best_probability is None or best_weight is None:
        raise RuntimeError("No blend candidate was evaluated.")
    expected_selection = pd.DataFrame(
        {
            "id": train.loc[
                folds["fold"].to_numpy() == roles.validation_fold, "id"
            ].to_numpy(),
            "target": target_indices(train)[
                folds["fold"].to_numpy() == roles.validation_fold
            ],
        }
    ).sort_values("id").reset_index(drop=True)
    if not identity.equals(expected_selection):
        raise ValueError(
            "Prediction files must contain every selection-fold id exactly once "
            "with targets matching the verified training data."
        )
    group_lookup = folds.set_index("id")["prompt_group"]
    groups = identity["id"].map(group_lookup)
    if groups.isna().any():
        raise ValueError("Fold file is missing validation ids.")

    metrics = {
        "selection_warning": (
            "Blend weight was selected on fold 7 and is now frozen. Fold 8 may "
            "calibrate probabilities but must not retune this weight; fold 9 remains untouched."
        ),
        "bootstrap_interpretation": (
            "Descriptive uncertainty on the same data used for grid selection; "
            "it does not correct for selection bias."
        ),
        "best_structural_weight": best_weight,
        "best_sparse_weight": 1.0 - best_weight,
        "structural": evaluate_probabilities(y, structural),
        "sparse": evaluate_probabilities(y, sparse_probability),
        "blend": evaluate_probabilities(y, best_probability),
        "bootstrap_blend_vs_sparse": grouped_bootstrap_delta(
            y,
            sparse_probability,
            best_probability,
            groups.to_numpy(),
            seed=args.seed,
        ),
        "grid": candidates,
        "provenance": {
            **build_provenance(
                PROJECT_ROOT,
                __file__,
                [args.split_config],
                args.folds,
                dataset_hashes,
            ),
            "inputs": {
                path_label(args.structural, PROJECT_ROOT): file_sha256(
                    args.structural
                ),
                path_label(args.sparse, PROJECT_ROOT): file_sha256(
                    args.sparse
                ),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = identity.copy()
    output[TARGET_COLUMNS[0]] = best_probability[:, 0]
    output[TARGET_COLUMNS[1]] = best_probability[:, 1]
    output[TARGET_COLUMNS[2]] = best_probability[:, 2]
    output.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "grid"},
            indent=2,
        )
    )
    print(f"\nSaved blend artifacts to {args.output_dir}")
    print(f"Saved comparable metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
