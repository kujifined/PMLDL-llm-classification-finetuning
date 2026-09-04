from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any


class ClearMLTracker:
    """Small, failure-tolerant adapter around one ClearML Task per run.

    Local experiment records are the source of truth. ClearML initialization,
    logging, or upload failures are exposed through ``state`` and warnings but
    do not discard a completed local run.
    """

    def __init__(
        self,
        *,
        run_id: str,
        experiment_config: dict[str, Any],
        project_config: dict[str, Any],
        task_class: Any | None = None,
    ) -> None:
        self.run_id = run_id
        self.experiment_config = copy.deepcopy(experiment_config)
        self.project_config = copy.deepcopy(project_config)
        self._task_class = task_class
        self._task: Any | None = None
        self._offline_enabled = False
        tracking = experiment_config["tracking"]
        self.backend = str(tracking["backend"])
        self.mode = str(tracking["mode"])
        self.state: dict[str, Any] = {
            "backend": self.backend,
            "mode": self.mode,
            "status": "not_started",
            "task_id": None,
            "error": None,
        }

    @property
    def enabled(self) -> bool:
        return self.backend == "clearml" and self.mode != "disabled"

    def _record_failure(self, action: str, exc: Exception) -> None:
        message = f"ClearML {action} failed: {type(exc).__name__}: {exc}"
        self.state["status"] = "unavailable"
        self.state["error"] = message
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _resolve_task_class(self) -> Any:
        if self._task_class is not None:
            return self._task_class
        from clearml import Task  # type: ignore[import-not-found]

        return Task

    def start(self) -> None:
        if self.backend == "none" or self.mode == "disabled":
            self.state["status"] = "disabled"
            return
        try:
            task_class = self._resolve_task_class()
            if self.mode == "offline":
                task_class.set_offline(True)
                self._offline_enabled = True

            shared_tracking = self.project_config["tracking"]
            tags = self._tags(shared_tracking.get("required_tags", []))
            tags.extend(str(tag) for tag in self.experiment_config.get("tags", []))
            task_name = str(
                shared_tracking.get("task_name_template", "{run_id}")
            ).format(
                run_id=self.run_id,
                experiment_id=self.experiment_config["experiment_id"],
                owner=self.experiment_config["owner"],
                seed=self.experiment_config["seed"],
            )
            self._task = task_class.init(
                project_name=str(shared_tracking["project_name"]),
                task_name=task_name,
                tags=list(dict.fromkeys(tags)),
                reuse_last_task_id=False,
            )
            self._task.connect(
                copy.deepcopy(self.experiment_config), name="experiment"
            )
            self.state["status"] = "active"
            self.state["task_id"] = getattr(self._task, "id", None)
        except Exception as exc:  # ClearML may fail on credentials or network.
            self._record_failure("initialization", exc)
            self._task = None

    def _tags(self, required_names: list[str]) -> list[str]:
        values = {
            "experiment_id": self.experiment_config["experiment_id"],
            "owner": self.experiment_config["owner"],
            "track": self.experiment_config["track"],
            "evaluation_role": self.experiment_config["evaluation_role"],
            "seed": self.experiment_config["seed"],
        }
        return [f"{name}:{values[name]}" for name in required_names]

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        namespace: str,
        step: int | None,
    ) -> None:
        if self._task is None:
            return
        try:
            logger = self._task.get_logger()
            for name, value in metrics.items():
                if step is None and hasattr(logger, "report_single_value"):
                    logger.report_single_value(
                        name=f"{namespace}/{name}", value=float(value)
                    )
                else:
                    logger.report_scalar(
                        title=namespace,
                        series=name,
                        value=float(value),
                        iteration=0 if step is None else int(step),
                    )
        except Exception as exc:
            self._record_failure("metric logging", exc)

    def upload_artifact(self, name: str, path: str | Path) -> None:
        if self._task is None:
            return
        try:
            self._task.upload_artifact(name=name, artifact_object=Path(path))
        except Exception as exc:
            self._record_failure("artifact upload", exc)

    def close(
        self,
        *,
        run_status: str = "completed",
        status_reason: str | None = None,
    ) -> None:
        if self._task is not None:
            try:
                if run_status == "failed" and hasattr(self._task, "mark_failed"):
                    self._task.mark_failed(
                        status_reason=status_reason or "ExperimentRun failed",
                        force=True,
                    )
                else:
                    self._task.close()
                if self.state["status"] == "active":
                    self.state["status"] = "closed"
            except Exception as exc:
                self._record_failure("shutdown", exc)
            finally:
                self._task = None
        if self._offline_enabled:
            try:
                self._resolve_task_class().set_offline(False)
            except Exception as exc:
                self._record_failure("offline reset", exc)
            finally:
                self._offline_enabled = False
