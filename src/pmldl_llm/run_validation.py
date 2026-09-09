from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .experiment import validate_experiment_config
from .metrics import validate_metrics_document


COMPLETED_STATUSES = {"completed", "failed", "aborted"}
RUN_STATUSES = COMPLETED_STATUSES | {"running"}
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class RunValidationReport:
    run_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    run: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        run_id = self.run.get("run_id") if self.run else None
        return {
            "run_directory": self.run_dir.as_posix(),
            "run_id": run_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _load_json_object(
    path: Path, report: RunValidationReport
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.errors.append(f"Missing required file: {path.name}.")
        return None
    except json.JSONDecodeError as exc:
        report.errors.append(f"Invalid JSON in {path.name}: {exc}.")
        return None
    if not isinstance(value, dict):
        report.errors.append(f"{path.name} must contain a JSON object.")
        return None
    return value


def _parse_timestamp(
    value: Any, location: str, report: RunValidationReport
) -> datetime | None:
    if not isinstance(value, str):
        report.errors.append(f"{location} must be an ISO-8601 timestamp.")
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        report.errors.append(f"{location} must be an ISO-8601 timestamp.")
        return None
    if parsed.tzinfo is None:
        report.errors.append(f"{location} must include a timezone.")
        return None
    return parsed


def _validate_run_metadata(
    run: dict[str, Any], report: RunValidationReport
) -> None:
    required = {
        "schema_version",
        "run_id",
        "experiment_id",
        "owner",
        "title",
        "status",
        "smoke_test",
        "evaluation_role",
        "seed",
        "started_at",
        "ended_at",
        "duration_seconds",
        "git",
        "config_source",
        "result_directory",
        "artifact_directory",
        "tracking",
        "error",
    }
    missing = sorted(required.difference(run))
    unknown = sorted(set(run).difference(required))
    if missing:
        report.errors.append("run.json is missing fields: " + ", ".join(missing) + ".")
        return
    if unknown:
        report.errors.append("run.json has unknown fields: " + ", ".join(unknown) + ".")
    if run["schema_version"] != 1:
        report.errors.append("run.schema_version must equal 1.")
    if not isinstance(run["run_id"], str) or not run["run_id"]:
        report.errors.append("run.run_id must be a non-empty string.")
    status = run["status"]
    if not isinstance(status, str) or status not in RUN_STATUSES:
        report.errors.append(f"run.status has unsupported value {status!r}.")
    if not isinstance(run["smoke_test"], bool):
        report.errors.append("run.smoke_test must be boolean.")
    seed = run["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        report.errors.append("run.seed must be a non-negative integer.")

    started = _parse_timestamp(run["started_at"], "run.started_at", report)
    ended: datetime | None = None
    if isinstance(status, str) and status in COMPLETED_STATUSES:
        ended = _parse_timestamp(run["ended_at"], "run.ended_at", report)
        duration = run["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            report.errors.append(
                "run.duration_seconds must be a finite non-negative number."
            )
        if started is not None and ended is not None and ended < started:
            report.errors.append("run.ended_at must not precede run.started_at.")
    elif status == "running":
        if run["ended_at"] is not None or run["duration_seconds"] is not None:
            report.errors.append(
                "A running run must have null ended_at and duration_seconds."
            )

    git = run["git"]
    if not isinstance(git, dict) or set(git) != {"commit", "dirty"}:
        report.errors.append("run.git must contain exactly commit and dirty.")
    else:
        commit = git["commit"]
        if commit is not None and (
            not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit)
        ):
            report.errors.append("run.git.commit must be null or a 40-character SHA.")
        if git["dirty"] is not None and not isinstance(git["dirty"], bool):
            report.errors.append("run.git.dirty must be boolean or null.")

    tracking = run["tracking"]
    expected_tracking = {"backend", "mode", "status", "task_id", "error"}
    if not isinstance(tracking, dict) or set(tracking) != expected_tracking:
        report.errors.append(
            "run.tracking must contain backend, mode, status, task_id, and error."
        )
    else:
        if not isinstance(tracking["backend"], str) or tracking[
            "backend"
        ] not in {"clearml", "none"}:
            report.errors.append("run.tracking.backend is unsupported.")
        if not isinstance(tracking["mode"], str) or tracking["mode"] not in {
            "online",
            "offline",
            "disabled",
        }:
            report.errors.append("run.tracking.mode is unsupported.")
        if not isinstance(tracking["status"], str) or tracking["status"] not in {
            "active",
            "closed",
            "disabled",
            "unavailable",
        }:
            report.errors.append("run.tracking.status is unsupported.")
        if tracking["task_id"] is not None and not isinstance(
            tracking["task_id"], str
        ):
            report.errors.append("run.tracking.task_id must be a string or null.")
        if tracking["error"] is not None and not isinstance(
            tracking["error"], str
        ):
            report.errors.append("run.tracking.error must be a string or null.")
        if (
            tracking["backend"] == "clearml"
            and tracking["status"] == "unavailable"
        ):
            report.warnings.append(
                "ClearML was unavailable; only local evidence exists."
            )

    if status == "completed" and run["error"] is not None:
        report.errors.append("A completed run must have null error.")
    if status in ("failed", "aborted") and not isinstance(
        run["error"], dict
    ):
        report.errors.append("A failed or aborted run must describe its error.")


def _validate_cross_file_identity(
    report: RunValidationReport,
    project_root: Path,
    artifact_root: Path,
    require_artifacts: bool,
) -> None:
    assert report.run is not None
    assert report.config is not None
    assert report.metrics is not None
    run = report.run
    config = report.config
    metrics = report.metrics

    if report.run_dir.name != run.get("run_id"):
        report.errors.append("Run directory name must equal run.run_id.")
    for field_name in ("experiment_id", "evaluation_role", "seed"):
        values = (run.get(field_name), config.get(field_name), metrics.get(field_name))
        if values[0] != values[1] or values[0] != values[2]:
            report.errors.append(
                f"{field_name} must match in config.json, run.json, and metrics.json."
            )
    if run.get("status") != metrics.get("status"):
        report.errors.append("status must match in run.json and metrics.json.")
    for field_name in ("owner", "title", "smoke_test"):
        if run.get(field_name) != config.get(field_name):
            report.errors.append(
                f"{field_name} must match in config.json and run.json."
            )

    expected_result = report.run_dir.resolve()
    recorded_result = (project_root / str(run.get("result_directory", ""))).resolve()
    if recorded_result != expected_result:
        report.errors.append("run.result_directory does not identify this directory.")

    artifact_directory = (project_root / str(run.get("artifact_directory", ""))).resolve()
    expected_artifact = (artifact_root / str(run.get("run_id", ""))).resolve()
    try:
        artifact_directory.relative_to(artifact_root)
    except ValueError:
        report.errors.append("run.artifact_directory escapes the configured artifact root.")
    if artifact_directory != expected_artifact:
        report.errors.append(
            "run.artifact_directory must identify artifacts/<run_id>."
        )
    if require_artifacts and not artifact_directory.is_dir():
        report.errors.append(f"Artifact directory does not exist: {artifact_directory}.")
    elif not artifact_directory.is_dir():
        report.warnings.append("Artifact directory is absent in this checkout.")


def validate_run_directory(
    run_dir: str | Path,
    *,
    project_root: str | Path,
    project_config_path: str | Path = "configs/project.json",
    require_artifacts: bool = False,
) -> RunValidationReport:
    root = Path(project_root).resolve()
    directory = Path(run_dir)
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    report = RunValidationReport(run_dir=directory)

    project_path = Path(project_config_path)
    if not project_path.is_absolute():
        project_path = root / project_path
    project = _load_json_object(project_path.resolve(), report)
    if project is None:
        return report

    report.config = _load_json_object(directory / "config.json", report)
    report.run = _load_json_object(directory / "run.json", report)
    report.metrics = _load_json_object(directory / "metrics.json", report)
    if report.config is None or report.run is None or report.metrics is None:
        return report

    try:
        validate_experiment_config(
            report.config, project, enforce_project_stage=False
        )
    except (KeyError, TypeError, ValueError) as exc:
        report.errors.append(f"config.json violates the experiment contract: {exc}")

    _validate_run_metadata(report.run, report)
    completed_full_run = (
        report.run.get("status") == "completed"
        and report.config.get("smoke_test") is False
    )
    if completed_full_run and project.get("run_policy", {}).get(
        "require_clean_git", False
    ):
        git = report.run.get("git")
        if not isinstance(git, dict) or git.get("commit") is None:
            report.errors.append("A completed full run must record a Git commit.")
        if not isinstance(git, dict) or git.get("dirty") is not False:
            report.errors.append("A completed full run must use a clean Git tree.")
    report.errors.extend(
        validate_metrics_document(
            report.metrics,
            required_metrics=project.get("required_metrics", []),
            require_required_metrics=completed_full_run,
        )
    )

    project_primary = project.get("primary_metric")
    metrics_primary = report.metrics.get("primary_metric")
    expected_primary = None
    if isinstance(project_primary, dict):
        expected_primary = {
            "name": project_primary.get("name"),
            "namespace": "validation",
            "direction": project_primary.get("direction"),
        }
    if metrics_primary != expected_primary:
        report.errors.append(
            "metrics.primary_metric does not match configs/project.json."
        )
    if completed_full_run and isinstance(metrics_primary, dict):
        namespace = metrics_primary.get("namespace")
        metric_name = metrics_primary.get("name")
        summary = report.metrics.get("summary")
        namespace_metrics = (
            summary.get(namespace)
            if isinstance(summary, dict) and isinstance(namespace, str)
            else None
        )
        if (
            not isinstance(namespace_metrics, dict)
            or not isinstance(metric_name, str)
            or metric_name not in namespace_metrics
        ):
            report.errors.append(
                "A completed full run must contain its primary metric value."
            )

    artifact_root = (
        root / str(project["results"]["large_artifact_directory"])
    ).resolve()
    _validate_cross_file_identity(
        report, root, artifact_root, require_artifacts
    )
    return report
