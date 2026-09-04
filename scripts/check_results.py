#!/usr/bin/env python3
"""Check that the baseline results table agrees with saved metric results."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = ("E000", "E001", "E002", "E010", "E011")
METRIC_COLUMNS = ("log_loss", "accuracy", "macro_f1")
DEFAULT_DISPLAY_TOLERANCE = 0.5e-6 + 1e-12


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Metric artifact does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metric artifact is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metric artifact must contain a JSON object: {path}")
    return value


def parse_results_table(path: Path) -> dict[str, tuple[str, dict[str, float]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"Results document does not exist: {path}") from exc

    rows: dict[str, tuple[str, dict[str, float]]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        match = re.match(r"^(E\d{3})\b", cells[0])
        if match is None or match.group(1) not in EXPERIMENT_IDS:
            continue
        experiment_id = match.group(1)
        if experiment_id in rows:
            raise ValueError(
                f"Duplicate {experiment_id} row in {path} (line {line_number})."
            )
        try:
            values = [float(cell) for cell in cells[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric metric in {experiment_id} row at {path}:{line_number}."
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Non-finite metric in {experiment_id} row at {path}:{line_number}."
            )
        rows[experiment_id] = (
            cells[0],
            dict(zip(METRIC_COLUMNS, values, strict=True)),
        )

    missing = [experiment_id for experiment_id in EXPERIMENT_IDS if experiment_id not in rows]
    if missing:
        raise ValueError(f"Missing results-table rows in {path}: {', '.join(missing)}")
    return rows


def saved_metrics(
    results_dir: Path,
    table_rows: dict[str, tuple[str, dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    bias = load_json(results_dir / "bias_baseline" / "evaluation.json")
    sparse = load_json(results_dir / "sparse_baseline" / "evaluation.json")
    blend = load_json(results_dir / "baseline_blend" / "evaluation.json")

    sparse_label = table_rows["E010"][0]
    alpha_match = re.search(
        r"\balpha\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        sparse_label,
        flags=re.IGNORECASE,
    )
    if alpha_match is None:
        sparse_metrics = sparse.get("best_validation")
    else:
        alpha_key = f"alpha={float(alpha_match.group(1)):g}"
        candidates = sparse.get("candidates")
        if not isinstance(candidates, dict) or alpha_key not in candidates:
            raise ValueError(
                f"{sparse_label!r} refers to {alpha_key}, which is absent from "
                f"{results_dir / 'sparse_baseline' / 'evaluation.json'}."
            )
        sparse_metrics = candidates[alpha_key]

    sources: dict[str, Any] = {
        "E000": bias.get("uniform"),
        "E001": bias.get("train_prior"),
        "E002": bias.get("bias_logistic"),
        "E010": sparse_metrics,
        "E011": blend.get("blend"),
    }
    for experiment_id, metrics in sources.items():
        if not isinstance(metrics, dict):
            raise ValueError(
                f"Missing metric object for {experiment_id} in saved results."
            )
        for metric_name in METRIC_COLUMNS:
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"Missing or non-finite {metric_name} for {experiment_id} "
                    "in saved results."
                )
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the E000-E011 values displayed in docs/RESULTS.md "
            "match the saved baseline metric results."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "docs" / "RESULTS.md",
        help="Path to the Markdown results document.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines",
        help="Directory containing the three versioned baseline result directories.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_DISPLAY_TOLERANCE,
        help="Absolute tolerance (default: half a unit at six decimal places).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be a finite non-negative number")

    try:
        rows = parse_results_table(args.results)
        expected = saved_metrics(args.results_dir, rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    mismatches = []
    for experiment_id in EXPERIMENT_IDS:
        displayed = rows[experiment_id][1]
        for metric_name in METRIC_COLUMNS:
            displayed_value = displayed[metric_name]
            artifact_value = float(expected[experiment_id][metric_name])
            difference = abs(displayed_value - artifact_value)
            if difference > args.tolerance:
                mismatches.append(
                    f"{experiment_id} {metric_name}: table={displayed_value:.9g}, "
                    f"artifact={artifact_value:.12g}, |delta|={difference:.3g}"
                )

    if mismatches:
        print("Results table does not match saved metrics:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"- {mismatch}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(EXPERIMENT_IDS)} experiment rows and "
        f"{len(EXPERIMENT_IDS) * len(METRIC_COLUMNS)} metrics match "
        f"within {args.tolerance:g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
