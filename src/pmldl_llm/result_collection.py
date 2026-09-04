from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .run_validation import RunValidationReport, validate_run_directory


METADATA_COLUMNS = [
    "rank",
    "run_id",
    "experiment_id",
    "owner",
    "title",
    "track",
    "parent_experiment_id",
    "changed_factor",
    "model_name",
    "model_revision",
    "seed",
    "evaluation_role",
    "smoke_test",
    "status",
    "primary_metric",
    "primary_metric_direction",
]
PROVENANCE_COLUMNS = [
    "git_commit",
    "git_dirty",
    "clearml_task_id",
    "tracking_status",
    "started_at",
    "ended_at",
    "duration_seconds",
]


@dataclass
class CollectionReport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    metric_columns: list[str] = field(default_factory=list)
    invalid_runs: list[RunValidationReport] = field(default_factory=list)
    excluded_counts: dict[str, int] = field(
        default_factory=lambda: {
            "not_completed": 0,
            "smoke_test": 0,
            "evaluation_role": 0,
        }
    )


def _load_project(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Project config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid project config JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Project config must contain an object.")
    return value


def _row_from_report(report: RunValidationReport) -> dict[str, Any]:
    assert report.run is not None
    assert report.config is not None
    assert report.metrics is not None
    run = report.run
    config = report.config
    metrics = report.metrics
    model = config["model"]
    git = run["git"]
    tracking = run["tracking"]
    validation_metrics = metrics["summary"].get("validation", {})
    return {
        "rank": None,
        "run_id": run["run_id"],
        "experiment_id": run["experiment_id"],
        "owner": run["owner"],
        "title": run["title"],
        "track": config["track"],
        "parent_experiment_id": config["parent_experiment_id"],
        "changed_factor": config["changed_factor"],
        "model_name": model["name"],
        "model_revision": model["revision"],
        "seed": run["seed"],
        "evaluation_role": run["evaluation_role"],
        "smoke_test": run["smoke_test"],
        "status": run["status"],
        "primary_metric": metrics["primary_metric"]["name"],
        "primary_metric_direction": metrics["primary_metric"]["direction"],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "clearml_task_id": tracking["task_id"],
        "tracking_status": tracking["status"],
        "started_at": run["started_at"],
        "ended_at": run["ended_at"],
        "duration_seconds": run["duration_seconds"],
        "_metrics": dict(validation_metrics),
    }


def _sort_rows(rows: list[dict[str, Any]]) -> None:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        metric_name = row["primary_metric"]
        direction = row["primary_metric_direction"]
        value = row["_metrics"].get(metric_name)
        missing = value is None
        sortable = 0.0 if missing else float(value)
        if direction == "maximize":
            sortable = -sortable
        return (
            missing,
            sortable,
            row["experiment_id"],
            row["seed"],
            row["run_id"],
        )

    rows.sort(key=key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank


def collect_result_rows(
    *,
    project_root: str | Path,
    project_config_path: str | Path = "configs/project.json",
    runs_dir: str | Path | None = None,
    evaluation_role: str | None = None,
    include_smoke: bool = False,
    require_artifacts: bool = False,
) -> CollectionReport:
    root = Path(project_root).resolve()
    project_path = Path(project_config_path)
    if not project_path.is_absolute():
        project_path = root / project_path
    project = _load_project(project_path.resolve())

    if runs_dir is None:
        run_root = root / project["results"]["run_directory"]
    else:
        run_root = Path(runs_dir)
        if not run_root.is_absolute():
            run_root = root / run_root
    run_root = run_root.resolve()
    selected_role = evaluation_role
    if selected_role is None:
        allowed_roles = project.get("allowed_evaluation_roles", [])
        if len(allowed_roles) != 1:
            raise ValueError(
                "Specify evaluation_role when the project allows multiple roles."
            )
        selected_role = allowed_roles[0]

    report = CollectionReport()
    if not run_root.exists():
        return report
    for directory in sorted(run_root.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        validation = validate_run_directory(
            directory,
            project_root=root,
            project_config_path=project_path,
            require_artifacts=require_artifacts,
        )
        if not validation.valid:
            report.invalid_runs.append(validation)
            continue
        assert validation.run is not None
        if validation.run["status"] != "completed":
            report.excluded_counts["not_completed"] += 1
            continue
        if validation.run["smoke_test"] and not include_smoke:
            report.excluded_counts["smoke_test"] += 1
            continue
        if validation.run["evaluation_role"] != selected_role:
            report.excluded_counts["evaluation_role"] += 1
            continue
        report.rows.append(_row_from_report(validation))

    required_metrics = list(project.get("required_metrics", []))
    observed_metrics = {
        name for row in report.rows for name in row["_metrics"]
    }
    report.metric_columns = required_metrics + sorted(
        observed_metrics.difference(required_metrics)
    )
    _sort_rows(report.rows)
    return report


def write_leaderboard(
    path: str | Path,
    collection: CollectionReport,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = METADATA_COLUMNS + collection.metric_columns + PROVENANCE_COLUMNS
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source_row in collection.rows:
            row = {key: value for key, value in source_row.items() if key != "_metrics"}
            row.update(source_row["_metrics"])
            writer.writerow(row)
        temporary = Path(handle.name)
    temporary.replace(output)
