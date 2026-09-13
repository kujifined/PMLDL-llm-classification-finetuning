#!/usr/bin/env python3
"""Validate three E2 QLoRA seed artifacts and build the fold-7 mean ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmldl_llm.constants import TARGET_COLUMNS
from pmldl_llm.evaluation import (
    evaluate_experiment_probabilities,
    normalize_probabilities,
)


ORIGINAL_COLUMNS = tuple(f"original_{name}" for name in TARGET_COLUMNS)
SWAPPED_COLUMNS = tuple(f"swapped_back_{name}" for name in TARGET_COLUMNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_member(artifact_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = artifact_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("class_order") != list(TARGET_COLUMNS):
        raise ValueError(f"Wrong class order in {manifest_path}")

    artifacts = manifest.get("artifacts", {})
    prediction_meta = artifacts.get("fold7_predictions", {})
    adapter_meta = artifacts.get("adapter", {})
    prediction_path = artifact_dir / str(prediction_meta.get("path", ""))
    adapter_path = artifact_dir / str(adapter_meta.get("path", ""))
    for path, metadata in (
        (prediction_path, prediction_meta),
        (adapter_path, adapter_meta),
    ):
        if not path.is_file():
            raise ValueError(f"Missing declared artifact: {path}")
        expected_hash = metadata.get("sha256")
        actual_hash = sha256(path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"SHA256 mismatch for {path}: {actual_hash} != {expected_hash}"
            )

    frame = pd.read_csv(prediction_path)
    required = {
        "id",
        "y_true",
        *ORIGINAL_COLUMNS,
        *SWAPPED_COLUMNS,
        *TARGET_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {prediction_path}: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate ids in {prediction_path}")
    for columns in (ORIGINAL_COLUMNS, SWAPPED_COLUMNS, TARGET_COLUMNS):
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
        normalized = normalize_probabilities(values)
        if not np.allclose(values, normalized, atol=1e-10, rtol=0.0):
            raise ValueError(f"Invalid probabilities in {prediction_path}: {columns}")
    return manifest, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        action="append",
        required=True,
        help="One completed seed artifact directory; pass exactly three times.",
    )
    parser.add_argument(
        "--result-output-dir",
        type=Path,
        default=ROOT / "results/e2_multiseed",
    )
    parser.add_argument(
        "--artifact-output-dir",
        type=Path,
        default=ROOT / "artifacts/e2_multiseed",
    )
    args = parser.parse_args()
    if len(args.artifact_dir) != 3:
        parser.error("--artifact-dir must be passed exactly three times")

    members = [load_member(path.resolve()) for path in args.artifact_dir]
    seeds = [int(manifest["seed"]) for manifest, _ in members]
    if len(set(seeds)) != 3 or 42 not in seeds:
        raise ValueError(f"Expected three distinct seeds including 42, got {seeds}")

    reference = members[0][1]
    reference_ids = reference["id"].to_numpy()
    reference_targets = reference["y_true"].to_numpy(dtype=np.int64)
    for manifest, frame in members[1:]:
        if not np.array_equal(reference_ids, frame["id"].to_numpy()):
            raise ValueError(f"Row-id mismatch for seed {manifest['seed']}")
        if not np.array_equal(
            reference_targets, frame["y_true"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"Target mismatch for seed {manifest['seed']}")

    rows: list[dict[str, object]] = []
    originals: list[np.ndarray] = []
    swapped: list[np.ndarray] = []
    for manifest, frame in members:
        original = frame.loc[:, list(ORIGINAL_COLUMNS)].to_numpy(dtype=np.float64)
        swapped_back = frame.loc[:, list(SWAPPED_COLUMNS)].to_numpy(dtype=np.float64)
        originals.append(original)
        swapped.append(swapped_back)
        metrics = evaluate_experiment_probabilities(
            reference_targets, original, swapped_back
        )
        rows.append(
            {
                "candidate": f"seed_{manifest['seed']}",
                "seed": int(manifest["seed"]),
                "run_id": str(manifest["run_id"]),
                **metrics,
            }
        )

    mean_original = normalize_probabilities(np.mean(originals, axis=0))
    mean_swapped = normalize_probabilities(np.mean(swapped, axis=0))
    mean_probabilities = normalize_probabilities(
        0.5 * (mean_original + mean_swapped)
    )
    ensemble_metrics = evaluate_experiment_probabilities(
        reference_targets, mean_original, mean_swapped
    )
    rows.append(
        {
            "candidate": "three_seed_mean",
            "seed": "17+42+73",
            "run_id": "+".join(str(manifest["run_id"]) for manifest, _ in members),
            **ensemble_metrics,
        }
    )

    result_output_dir = args.result_output_dir.resolve()
    artifact_output_dir = args.artifact_output_dir.resolve()
    result_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(rows).sort_values("log_loss", kind="stable")
    comparison_path = result_output_dir / "fold7_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    ensemble_prediction_path = artifact_output_dir / "fold7_predictions.csv"
    ensemble_frame = pd.DataFrame(
        {
            "id": reference_ids,
            "y_true": reference_targets,
            **{
                f"original_{name}": mean_original[:, index]
                for index, name in enumerate(TARGET_COLUMNS)
            },
            **{
                f"swapped_back_{name}": mean_swapped[:, index]
                for index, name in enumerate(TARGET_COLUMNS)
            },
            **{
                name: mean_probabilities[:, index]
                for index, name in enumerate(TARGET_COLUMNS)
            },
        }
    )
    ensemble_frame.to_csv(ensemble_prediction_path, index=False)

    winner = comparison.iloc[0]
    member_losses = comparison.loc[
        comparison["candidate"] != "three_seed_mean", "log_loss"
    ]
    summary = {
        "schema_version": 1,
        "evaluation_role": "selection",
        "selection_fold": 7,
        "class_order": list(TARGET_COLUMNS),
        "members": [
            {
                "experiment_id": manifest["experiment_id"],
                "run_id": manifest["run_id"],
                "seed": manifest["seed"],
                "adapter_sha256": manifest["artifacts"]["adapter"]["sha256"],
                "fold7_predictions_sha256": manifest["artifacts"]
                ["fold7_predictions"]["sha256"],
            }
            for manifest, _ in members
        ],
        "three_seed_mean": ensemble_metrics,
        "best_single_log_loss": float(member_losses.min()),
        "mean_member_log_loss": float(member_losses.mean()),
        "ensemble_delta_vs_best_single": float(
            ensemble_metrics["log_loss"] - member_losses.min()
        ),
        "recommended_candidate": str(winner["candidate"]),
        "recommendation_log_loss": float(winner["log_loss"]),
        "fold8_status": "unopened",
        "kaggle_used_for_selection": False,
    }
    summary_path = result_output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    ensemble_manifest = {
        "schema_version": 1,
        "candidate": "three_seed_mean",
        "class_order": list(TARGET_COLUMNS),
        "members": summary["members"],
        "fold7_predictions": {
            "path": "fold7_predictions.csv",
            "sha256": sha256(ensemble_prediction_path),
            "bytes": ensemble_prediction_path.stat().st_size,
            "rows": len(ensemble_frame),
        },
        "metrics": ensemble_metrics,
        "selection": {
            "recommended_candidate": summary["recommended_candidate"],
            "fold8_status": "unopened",
            "kaggle_used": False,
        },
    }
    ensemble_manifest_path = artifact_output_dir / "model_manifest.json"
    ensemble_manifest_path.write_text(
        json.dumps(
            ensemble_manifest, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Handoff manifest: {ensemble_manifest_path}")


if __name__ == "__main__":
    main()
