#!/usr/bin/env python3
"""Run the existing E002 baseline inside the canonical ExperimentRun contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.constants import DEFAULT_DATA_DIR, TARGET_COLUMNS
from pmldl_llm.experiment import ExperimentRun


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def brier_score_from_predictions(path: Path) -> float:
    frame = pd.read_csv(path)
    required = {"target", *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Validation predictions are missing columns: {', '.join(missing)}"
        )
    targets = frame["target"].to_numpy(dtype=np.int64)
    if len(targets) == 0 or not np.isin(targets, [0, 1, 2]).all():
        raise ValueError("Validation targets must be non-empty class indices 0, 1, 2.")
    probabilities = frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities must be finite.")
    one_hot = np.eye(3, dtype=np.float64)[targets]
    return float(np.square(probabilities - one_hot).sum(axis=1).mean())


def canonical_metrics(evaluation: dict[str, Any], predictions_path: Path) -> dict[str, float]:
    baseline = evaluation.get("bias_logistic")
    if not isinstance(baseline, dict):
        raise ValueError("Baseline evaluation has no bias_logistic metric object.")
    scalar_names = ("log_loss", "accuracy", "macro_f1", "ece_15")
    metrics: dict[str, float] = {}
    for name in scalar_names:
        value = baseline.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Baseline evaluation has no numeric {name} metric.")
        metrics[name] = float(value)
    symmetry = evaluation.get("raw_symmetry_l1")
    if isinstance(symmetry, bool) or not isinstance(symmetry, (int, float)):
        raise ValueError("Baseline evaluation has no numeric raw_symmetry_l1 metric.")
    metrics["brier_score"] = brier_score_from_predictions(predictions_path)
    metrics["swap_error_l1"] = float(symmetry)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/E002.json"),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    with ExperimentRun(args.experiment_config, project_root=PROJECT_ROOT) as run:
        baseline_dir = run.artifact_path("baseline")
        evaluation_path = baseline_dir / "evaluation.json"
        log_path = run.artifact_path("baseline/training.log")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_bias_baseline.py"),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(baseline_dir),
            "--evaluation-output",
            str(evaluation_path),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="")
        run.log_artifact("baseline/training.log", log_path)
        if completed.returncode != 0:
            raise RuntimeError(
                "E002 baseline process failed with exit code "
                f"{completed.returncode}; see {log_path}."
            )

        evaluation = load_json_object(evaluation_path)
        predictions_path = baseline_dir / "validation_predictions.csv"
        run.log_metrics(canonical_metrics(evaluation, predictions_path))
        for name in (
            "evaluation.json",
            "evaluation_model.joblib",
            "full_model.joblib",
            "feature_coefficients.csv",
            "validation_predictions.csv",
            "test_schema_predictions.csv",
        ):
            run.log_artifact(f"baseline/{name}", baseline_dir / name)

    print(f"Completed run: {run.run_id}")
    print(f"Versioned result: {run.result_dir}")
    print(f"Ignored artifacts: {run.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
