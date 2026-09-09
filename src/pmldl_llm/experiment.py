from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from types import TracebackType
from typing import Any

from .metrics import (
    METRIC_NAME_PATTERN,
    METRICS_SCHEMA_VERSION,
    validate_metrics_document,
)
from .provenance import git_state
from .tracking import ClearMLTracker


EXPERIMENT_ID_PATTERN = re.compile(r"^E[0-9]{3,}$")
ALLOWED_TRACKS = {
    "sanity",
    "classical",
    "neural",
    "ablation",
    "calibration",
    "ensemble",
}
ALLOWED_STATUSES = {"planned", "running", "completed", "failed", "aborted"}
ALLOWED_EVALUATION_ROLES = {"selection", "calibration", "final_holdout"}
ALLOWED_TRACKING_BACKENDS = {"clearml", "none"}
ALLOWED_TRACKING_MODES = {"online", "offline", "disabled"}
REQUIRED_EXPERIMENT_FIELDS = {
    "schema_version",
    "experiment_id",
    "owner",
    "title",
    "track",
    "status",
    "hypothesis",
    "parent_experiment_id",
    "changed_factor",
    "seed",
    "evaluation_role",
    "smoke_test",
    "model",
    "training",
    "tracking",
    "tags",
}
OPTIONAL_EXPERIMENT_FIELDS = {"$schema", "notes"}


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def validate_experiment_config(
    experiment: dict[str, Any],
    project: dict[str, Any],
    *,
    enforce_project_stage: bool = True,
) -> None:
    missing = sorted(REQUIRED_EXPERIMENT_FIELDS.difference(experiment))
    unknown = sorted(
        set(experiment).difference(
            REQUIRED_EXPERIMENT_FIELDS | OPTIONAL_EXPERIMENT_FIELDS
        )
    )
    if missing:
        raise ValueError(f"Experiment config is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Experiment config has unknown fields: {', '.join(unknown)}")
    if experiment["schema_version"] != project.get("schema_version"):
        raise ValueError("Experiment and project schema versions do not match.")
    experiment_id = experiment["experiment_id"]
    if not isinstance(experiment_id, str) or not EXPERIMENT_ID_PATTERN.fullmatch(
        experiment_id
    ):
        raise ValueError("experiment_id must match E followed by at least three digits.")
    for name, minimum in (
        ("owner", 1),
        ("title", 3),
        ("hypothesis", 10),
        ("changed_factor", 3),
    ):
        if (
            not isinstance(experiment[name], str)
            or len(experiment[name].strip()) < minimum
        ):
            raise ValueError(f"{name} must contain at least {minimum} characters.")
    if experiment["track"] not in ALLOWED_TRACKS:
        raise ValueError(f"Unsupported experiment track: {experiment['track']}")
    if experiment["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported experiment status: {experiment['status']}")
    parent = experiment["parent_experiment_id"]
    if parent is not None and (
        not isinstance(parent, str) or not EXPERIMENT_ID_PATTERN.fullmatch(parent)
    ):
        raise ValueError("parent_experiment_id must be null or a valid experiment ID.")
    seed = experiment["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    role = experiment["evaluation_role"]
    if role not in ALLOWED_EVALUATION_ROLES:
        raise ValueError(f"Unsupported evaluation_role: {role}")
    if enforce_project_stage and role not in project.get(
        "allowed_evaluation_roles", []
    ):
        raise ValueError(
            f"evaluation_role={role!r} is locked by configs/project.json."
        )
    if not isinstance(experiment["smoke_test"], bool):
        raise ValueError("smoke_test must be boolean.")
    model = experiment["model"]
    if not isinstance(model, dict) or set(model) != {"name", "revision"}:
        raise ValueError("model must contain exactly name and revision.")
    if any(
        value is not None and not isinstance(value, str)
        for value in model.values()
    ):
        raise ValueError("model name and revision must be strings or null.")
    if not isinstance(experiment["training"], dict):
        raise ValueError("training must be a JSON object.")
    tracking = experiment["tracking"]
    if not isinstance(tracking, dict) or set(tracking) != {"backend", "mode"}:
        raise ValueError("tracking must contain exactly backend and mode.")
    if tracking["backend"] not in ALLOWED_TRACKING_BACKENDS:
        raise ValueError(f"Unsupported tracking backend: {tracking['backend']}")
    if tracking["mode"] not in ALLOWED_TRACKING_MODES:
        raise ValueError(f"Unsupported tracking mode: {tracking['mode']}")
    tags = experiment["tags"]
    if (
        not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        or len(tags) != len(set(tags))
    ):
        raise ValueError("tags must contain unique, non-empty strings.")
    try:
        json.dumps(experiment, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Experiment config must be finite and JSON serializable.") from exc


class ExperimentRun:
    """Context manager for a reproducible local run and optional ClearML task."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        project_root: str | Path | None = None,
        project_config_path: str | Path | None = None,
        tracker_class: type[ClearMLTracker] = ClearMLTracker,
        task_class: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.config_path = self._resolve(config_path)
        self.project_config_path = self._resolve(
            project_config_path or "configs/project.json"
        )
        self.config = _load_object(self.config_path)
        self.project_config = _load_object(self.project_config_path)
        validate_experiment_config(self.config, self.project_config)
        self._validate_policy()

        git = git_state(self.project_root)
        self.git = git
        if not self.config["smoke_test"] and self.project_config["run_policy"].get(
            "require_clean_git", False
        ):
            if git["commit"] is None or git["dirty"] is not False:
                raise RuntimeError("A full experiment requires a clean Git commit.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        revision = str(git["commit"])[:8] if git["commit"] else "nogit"
        self.run_id = (
            f"{self.config['experiment_id']}__s{self.config['seed']}__"
            f"{revision}__{timestamp}"
        )
        results = self.project_config["results"]
        self.result_dir = self._resolve(results["run_directory"]) / self.run_id
        self.artifact_dir = (
            self._resolve(results["large_artifact_directory"]) / self.run_id
        )
        self._run_path = self.result_dir / "run.json"
        self._metrics_path = self.result_dir / "metrics.json"
        self._started_monotonic: float | None = None
        self._started_at: str | None = None
        self._closed = False
        self._metrics: dict[str, Any] = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment_id": self.config["experiment_id"],
            "status": "running",
            "seed": self.config["seed"],
            "evaluation_role": self.config["evaluation_role"],
            "primary_metric": {
                "name": self.project_config["primary_metric"]["name"],
                "namespace": "validation",
                "direction": self.project_config["primary_metric"]["direction"],
            },
            "summary": {},
            "history": [],
        }
        self.tracker = tracker_class(
            run_id=self.run_id,
            experiment_config=self.config,
            project_config=self.project_config,
            task_class=task_class,
        )

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        if value.is_absolute():
            return value.resolve()
        return (self.project_root / value).resolve()

    def _validate_policy(self) -> None:
        policy = self.project_config.get("run_policy", {})
        if (
            not self.config["smoke_test"]
            and policy.get("require_parent_for_controlled_experiment", False)
            and self.config["track"] in {"ablation", "calibration", "ensemble"}
            and self.config["parent_experiment_id"] is None
        ):
            raise ValueError(
                "A full controlled experiment requires parent_experiment_id."
            )

    def __enter__(self) -> "ExperimentRun":
        self.result_dir.mkdir(parents=True, exist_ok=False)
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._started_monotonic = time.perf_counter()
        _atomic_json(self.result_dir / "config.json", self.config)
        self.tracker.start()
        self._write_run(status="running")
        self._write_metrics()
        return self

    def _write_run(
        self,
        *,
        status: str,
        ended_at: str | None = None,
        duration_seconds: float | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "experiment_id": self.config["experiment_id"],
            "owner": self.config["owner"],
            "title": self.config["title"],
            "status": status,
            "smoke_test": self.config["smoke_test"],
            "evaluation_role": self.config["evaluation_role"],
            "seed": self.config["seed"],
            "started_at": self._started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "git": self.git,
            "config_source": self._relative_label(self.config_path),
            "result_directory": self._relative_label(self.result_dir),
            "artifact_directory": self._relative_label(self.artifact_dir),
            "tracking": dict(self.tracker.state),
            "error": error,
        }
        _atomic_json(self._run_path, payload)

    def _relative_label(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _write_metrics(self) -> None:
        _atomic_json(self._metrics_path, self._metrics)

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        *,
        namespace: str = "validation",
        step: int | None = None,
    ) -> None:
        if self._started_monotonic is None or self._closed:
            raise RuntimeError(
                "Metrics can only be logged inside an active ExperimentRun."
            )
        if not METRIC_NAME_PATTERN.fullmatch(namespace):
            raise ValueError("Metric namespace must use lowercase snake_case.")
        clean: dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not METRIC_NAME_PATTERN.fullmatch(name):
                raise ValueError("Metric names must use lowercase snake_case.")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {name!r} must be numeric.")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Metric {name!r} must be finite.")
            clean[name] = numeric
        summary = self._metrics["summary"].setdefault(namespace, {})
        summary.update(clean)
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("Metric step must be a non-negative integer.")
            self._metrics["history"].append(
                {"namespace": namespace, "step": step, "metrics": clean}
            )
        self._write_metrics()
        self.tracker.log_metrics(clean, namespace=namespace, step=step)

    def log_metric(
        self,
        name: str,
        value: int | float,
        *,
        namespace: str = "validation",
        step: int | None = None,
    ) -> None:
        self.log_metrics({name: value}, namespace=namespace, step=step)

    def artifact_path(self, relative_path: str | Path) -> Path:
        path = (self.artifact_dir / relative_path).resolve()
        try:
            path.relative_to(self.artifact_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "Artifact path must stay inside this run directory."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_artifact(self, name: str, source: str | Path) -> Path:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise ValueError(
                f"Artifact source must be an existing file: {source_path}"
            )
        target = self.artifact_path(name)
        if source_path != target:
            shutil.copy2(source_path, target)
        self.tracker.upload_artifact(name, target)
        return target

    def _required_metrics_missing(self) -> list[str]:
        validation = self._metrics["summary"].get("validation", {})
        required = list(self.project_config.get("required_metrics", []))
        primary = self.project_config["primary_metric"]["name"]
        if primary not in required:
            required.append(primary)
        return [
            name
            for name in required
            if name not in validation
        ]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        if self._closed:
            return False
        assert self._started_monotonic is not None
        duration = float(time.perf_counter() - self._started_monotonic)
        validation = self._metrics["summary"].setdefault("validation", {})
        validation["runtime_seconds"] = duration

        completion_error = exc_value
        if completion_error is None and not self.config["smoke_test"]:
            missing = self._required_metrics_missing()
            if missing:
                completion_error = RuntimeError(
                    "Full run is missing required validation metrics: "
                    + ", ".join(missing)
                )

        self._metrics["status"] = "failed" if completion_error else "completed"
        contract_errors = validate_metrics_document(self._metrics)
        if contract_errors and completion_error is None:
            completion_error = RuntimeError(
                "Generated metrics.json violates its contract: "
                + " ".join(contract_errors)
            )
            self._metrics["status"] = "failed"
        self._write_metrics()
        self.tracker.log_metrics(
            {"runtime_seconds": float(validation["runtime_seconds"])},
            namespace="validation",
            step=None,
        )
        self.tracker.close(
            run_status="failed" if completion_error else "completed",
            status_reason=str(completion_error) if completion_error else None,
        )

        error = None
        if completion_error is not None:
            error = {
                "type": type(completion_error).__name__,
                "message": str(completion_error),
            }
            if exc_type is not None:
                error["traceback"] = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
        self._write_run(
            status="failed" if completion_error else "completed",
            ended_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            error=error,
        )
        self._closed = True
        if exc_value is None and completion_error is not None:
            raise completion_error
        return False
