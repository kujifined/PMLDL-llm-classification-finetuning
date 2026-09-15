#!/usr/bin/env python3
"""Select fold-7 candidates and fit a bounded fold-8 calibrated blend."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, log_loss


CLASS_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")
EPSILON = 1e-15
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PredictionSet:
    name: str
    path: Path
    checkpoint: str
    frame: pd.DataFrame
    probabilities: np.ndarray
    targets: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("select", "blend", "apply"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--final-model-output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "final_model.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_COLUMNS):
        raise ValueError("Probabilities must have shape (rows, 3).")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every probability row must have positive mass.")
    return values / totals


def load_prediction_set(spec: Mapping[str, object], role: str) -> PredictionSet:
    name = str(spec["name"])
    raw_path = spec.get(f"{role}_predictions")
    if not raw_path:
        raise ValueError(f"{name} has no {role}_predictions path.")
    path = Path(str(raw_path))
    frame = pd.read_csv(path)
    if role != "inference" and "target" not in frame and "y_true" in frame:
        frame = frame.rename(columns={"y_true": "target"})
    required = {"id", *CLASS_COLUMNS}
    if role != "inference":
        required.add("target")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{path} must contain unique, non-null ids.")
    probabilities = normalize(frame.loc[:, CLASS_COLUMNS].to_numpy())
    if role == "inference":
        targets = np.empty(0, dtype=np.int64)
    else:
        targets = frame["target"].to_numpy(dtype=np.int64)
        if not np.isin(targets, np.arange(len(CLASS_COLUMNS))).all():
            raise ValueError(f"{path} target must contain only 0, 1, or 2.")
    return PredictionSet(
        name=name,
        path=path,
        checkpoint=str(spec["checkpoint"]),
        frame=frame,
        probabilities=probabilities,
        targets=targets,
    )


def align_prediction_sets(values: Sequence[PredictionSet], role: str) -> None:
    if not values:
        raise ValueError("At least one prediction set is required.")
    reference_ids = values[0].frame["id"].to_numpy()
    reference_targets = values[0].targets
    for value in values[1:]:
        if not np.array_equal(reference_ids, value.frame["id"].to_numpy()):
            raise ValueError(f"{value.name} ids or row order do not match.")
        if role != "inference" and not np.array_equal(
            reference_targets, value.targets
        ):
            raise ValueError(f"{value.name} targets do not match.")


def metric_row(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 16)
    for index in range(len(bin_edges) - 1):
        lower, upper = bin_edges[index : index + 2]
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    one_hot = np.eye(len(CLASS_COLUMNS), dtype=np.float64)[targets]
    return {
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "ece_15": ece,
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def calibrate(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    logits = np.log(np.clip(probabilities, EPSILON, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return normalize(values)


def fit_temperature(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, np.ndarray]:
    result = minimize_scalar(
        lambda value: log_loss(
            targets,
            calibrate(probabilities, float(value)),
            labels=[0, 1, 2],
        ),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(result.x)
    return temperature, calibrate(probabilities, temperature)


def blend(values: Sequence[PredictionSet], weights: Sequence[float]) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=np.float64)
    if len(values) != len(weight_array) or (weight_array <= 0).any():
        raise ValueError("Blend weights must be positive and match model count.")
    if not np.isclose(weight_array.sum(), 1.0, atol=1e-12):
        raise ValueError("Blend weights must sum to one.")
    return normalize(
        sum(weight * value.probabilities for weight, value in zip(weight_array, values))
    )


def candidate_weights(model_count: int, grid: Sequence[float]) -> list[tuple[float, ...]]:
    if model_count not in (2, 3):
        raise ValueError("Only pair and triple blends are supported.")
    candidates: set[tuple[float, ...]] = {
        tuple([1.0 / model_count] * model_count)
    }
    for raw in itertools.product(grid, repeat=model_count):
        weights = tuple(float(value) for value in raw)
        if all(value > 0 for value in weights) and np.isclose(sum(weights), 1.0):
            candidates.add(weights)
    return sorted(candidates)


def select_stage(config: Mapping[str, object], output_dir: Path) -> None:
    policy = config["selection_policy"]
    threshold = float(policy["max_absolute_log_loss_delta"])
    anchor_name = str(policy["anchor"])
    values = [
        load_prediction_set(spec, "fold7") for spec in config["candidates"]
    ]
    align_prediction_sets(values, "fold7")
    rows = []
    for value in values:
        row = {"candidate": value.name, **metric_row(value.targets, value.probabilities)}
        row.update(
            {
                "checkpoint": value.checkpoint,
                "predictions_sha256": sha256_file(value.path),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    anchor_rows = table.loc[table["candidate"] == anchor_name, "log_loss"]
    if len(anchor_rows) != 1:
        raise ValueError("Selection anchor must identify exactly one candidate.")
    anchor_loss = float(anchor_rows.iloc[0])
    table["delta_vs_anchor"] = table["log_loss"] - anchor_loss
    table["selected_for_fold8"] = table["delta_vs_anchor"] <= threshold
    table["selection_rule"] = f"delta_vs_{anchor_name} <= {threshold:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["log_loss", "candidate"]).to_csv(
        output_dir / "single_model_selection.csv", index=False
    )


def blend_stage(
    config: Mapping[str, object], output_dir: Path, final_model_output: Path
) -> None:
    selected_names = set(config["selected_candidates"])
    values = [
        load_prediction_set(spec, "fold8")
        for spec in config["candidates"]
        if spec["name"] in selected_names
    ]
    if {value.name for value in values} != selected_names:
        raise ValueError("Every selected candidate needs fold8 predictions.")
    align_prediction_sets(values, "fold8")
    bounds = tuple(
        float(value) for value in config["calibration"]["temperature_bounds"]
    )
    grid = tuple(float(value) for value in config["blend_search"]["weight_grid"])
    rows: list[dict[str, object]] = []
    predictions_by_name: dict[str, np.ndarray] = {}
    for count in (2, 3):
        for members in itertools.combinations(values, count):
            for weights in candidate_weights(count, grid):
                uncalibrated = blend(members, weights)
                temperature, calibrated = fit_temperature(
                    members[0].targets, uncalibrated, bounds
                )
                name = "+".join(value.name for value in members)
                key = name + "@" + "+".join(f"{value:.6f}" for value in weights)
                predictions_by_name[key] = calibrated
                rows.append(
                    {
                        "blend_id": key,
                        "models": name,
                        "weights": "+".join(f"{value:.6f}" for value in weights),
                        "model_count": count,
                        **{
                            f"uncalibrated_{metric}": value
                            for metric, value in metric_row(
                                members[0].targets, uncalibrated
                            ).items()
                        },
                        "temperature": temperature,
                        **{
                            f"calibrated_{metric}": value
                            for metric, value in metric_row(
                                members[0].targets, calibrated
                            ).items()
                        },
                    }
                )
    grid_frame = pd.DataFrame(rows).sort_values(
        ["calibrated_log_loss", "model_count", "blend_id"]
    )
    winner = grid_frame.iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_frame.to_csv(output_dir / "ensemble_grid.csv", index=False)
    comparison_rows: list[dict[str, object]] = []
    for value in values:
        temperature, calibrated = fit_temperature(
            value.targets, value.probabilities, bounds
        )
        comparison_rows.append(
            {
                "candidate": value.name,
                "kind": "single",
                "models": value.name,
                "weights": "1.000000",
                "temperature": temperature,
                **metric_row(value.targets, calibrated),
            }
        )
    comparison_rows.append(
        {
            "candidate": str(winner["blend_id"]),
            "kind": "blend",
            "models": str(winner["models"]),
            "weights": str(winner["weights"]),
            "temperature": float(winner["temperature"]),
            **{
                metric: float(winner[f"calibrated_{metric}"])
                for metric in (
                    "log_loss",
                    "accuracy",
                    "macro_f1",
                    "ece_15",
                    "brier_score",
                )
            },
        }
    )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["log_loss", "kind", "candidate"]
    )
    comparison["selected"] = comparison["candidate"] == str(winner["blend_id"])
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)
    winner_models = str(winner["models"]).split("+")
    winner_weights = [float(value) for value in str(winner["weights"]).split("+")]
    by_name = {value.name: value for value in values}
    frozen = {
        "schema_version": 1,
        "status": "frozen",
        "class_order": list(CLASS_COLUMNS),
        "selection_protocol": {
            "candidate_fold": 7,
            "blend_and_calibration_fold": 8,
            "public_leaderboard_used": False,
        },
        "models": [
            {
                "name": name,
                "weight": weight,
                "checkpoint": by_name[name].checkpoint,
                "fold8_predictions_sha256": sha256_file(by_name[name].path),
            }
            for name, weight in zip(winner_models, winner_weights)
        ],
        "calibration": {
            "method": "multiclass_temperature_scaling",
            "temperature": float(winner["temperature"]),
            "fitted_on_fold": 8,
        },
        "fold8_metrics": {
            "log_loss": float(winner["calibrated_log_loss"]),
            "accuracy": float(winner["calibrated_accuracy"]),
            "macro_f1": float(winner["calibrated_macro_f1"]),
        },
    }
    final_model_output.parent.mkdir(parents=True, exist_ok=True)
    final_model_output.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    final_probabilities = predictions_by_name[str(winner["blend_id"])]
    pd.DataFrame(
        {
            "id": values[0].frame["id"].to_numpy(),
            "target": values[0].targets,
            **{
                name: final_probabilities[:, index]
                for index, name in enumerate(CLASS_COLUMNS)
            },
        }
    ).to_csv(output_dir / "fold8_final_probabilities.csv", index=False)


def apply_stage(config: Mapping[str, object], output_dir: Path) -> None:
    frozen_path = Path(str(config["frozen_model"]))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    weights = {value["name"]: float(value["weight"]) for value in frozen["models"]}
    values = [
        load_prediction_set(spec, "inference")
        for spec in config["candidates"]
        if spec["name"] in weights
    ]
    align_prediction_sets(values, "inference")
    probabilities = blend(values, [weights[value.name] for value in values])
    probabilities = calibrate(
        probabilities, float(frozen["calibration"]["temperature"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": values[0].frame["id"].to_numpy(),
            **{
                name: probabilities[:, index]
                for index, name in enumerate(CLASS_COLUMNS)
            },
        }
    ).to_csv(output_dir / "final_probabilities.csv", index=False)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "select":
        select_stage(config, args.output_dir)
    elif args.stage == "blend":
        blend_stage(config, args.output_dir, args.final_model_output)
    else:
        apply_stage(config, args.output_dir)


if __name__ == "__main__":
    main()
