from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pmldl_llm.tracking import ClearMLTracker


class FakeLogger:
    def __init__(self) -> None:
        self.single_values: list[tuple[str, float]] = []
        self.scalars: list[tuple[str, str, float, int]] = []

    def report_single_value(self, *, name: str, value: float) -> None:
        self.single_values.append((name, value))

    def report_scalar(
        self, *, title: str, series: str, value: float, iteration: int
    ) -> None:
        self.scalars.append((title, series, value, iteration))


class FakeTaskInstance:
    id = "task-123"

    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.connected: tuple[dict[str, object], str] | None = None
        self.artifacts: list[tuple[str, Path]] = []
        self.closed = False

    def connect(self, config: dict[str, object], *, name: str) -> None:
        self.connected = (config, name)

    def get_logger(self) -> FakeLogger:
        return self.logger

    def upload_artifact(self, *, name: str, artifact_object: Path) -> None:
        self.artifacts.append((name, artifact_object))

    def close(self) -> None:
        self.closed = True


class FakeTask:
    offline_calls: list[bool] = []
    init_kwargs: dict[str, object] | None = None
    instance: FakeTaskInstance | None = None

    @classmethod
    def reset(cls) -> None:
        cls.offline_calls = []
        cls.init_kwargs = None
        cls.instance = None

    @classmethod
    def set_offline(cls, value: bool) -> None:
        cls.offline_calls.append(value)

    @classmethod
    def init(cls, **kwargs: object) -> FakeTaskInstance:
        cls.init_kwargs = kwargs
        cls.instance = FakeTaskInstance()
        return cls.instance


def experiment_tracking(mode: str = "online") -> dict[str, object]:
    return {
        "experiment_id": "E030",
        "owner": "Danil",
        "track": "neural",
        "evaluation_role": "selection",
        "seed": 42,
        "tracking": {"backend": "clearml", "mode": mode},
        "tags": ["candidate"],
    }


def project_tracking() -> dict[str, object]:
    return {
        "tracking": {
            "project_name": "PMLDL Test",
            "task_name_template": "{experiment_id}/{run_id}",
            "required_tags": [
                "experiment_id",
                "owner",
                "track",
                "evaluation_role",
                "seed",
            ],
        }
    }


class ClearMLTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeTask.reset()

    def test_wrapper_creates_one_tagged_task_and_logs_values(self) -> None:
        tracker = ClearMLTracker(
            run_id="E030__s42__abc__time",
            experiment_config=experiment_tracking(),
            project_config=project_tracking(),
            task_class=FakeTask,
        )
        tracker.start()
        tracker.log_metrics({"log_loss": 1.02}, namespace="validation", step=None)
        tracker.log_metrics({"loss": 0.5}, namespace="train", step=3)
        with tempfile.NamedTemporaryFile() as artifact:
            tracker.upload_artifact("model", artifact.name)
        tracker.close()

        self.assertEqual(tracker.state["task_id"], "task-123")
        self.assertEqual(tracker.state["status"], "closed")
        self.assertEqual(FakeTask.init_kwargs["project_name"], "PMLDL Test")
        self.assertEqual(
            FakeTask.init_kwargs["task_name"],
            "E030/E030__s42__abc__time",
        )
        self.assertIn("experiment_id:E030", FakeTask.init_kwargs["tags"])
        self.assertIn("candidate", FakeTask.init_kwargs["tags"])
        assert FakeTask.instance is not None
        self.assertEqual(
            FakeTask.instance.logger.single_values,
            [("validation/log_loss", 1.02)],
        )
        self.assertEqual(
            FakeTask.instance.logger.scalars,
            [("train", "loss", 0.5, 3)],
        )
        self.assertTrue(FakeTask.instance.closed)

    def test_offline_mode_is_reset_after_task_closes(self) -> None:
        tracker = ClearMLTracker(
            run_id="offline-run",
            experiment_config=experiment_tracking(mode="offline"),
            project_config=project_tracking(),
            task_class=FakeTask,
        )
        tracker.start()
        tracker.close()
        self.assertEqual(FakeTask.offline_calls, [True, False])

    def test_disabled_mode_does_not_initialize_clearml(self) -> None:
        config = experiment_tracking(mode="disabled")
        tracker = ClearMLTracker(
            run_id="disabled-run",
            experiment_config=config,
            project_config=project_tracking(),
            task_class=FakeTask,
        )
        tracker.start()
        self.assertEqual(tracker.state["status"], "disabled")
        self.assertIsNone(FakeTask.init_kwargs)

    def test_initialization_failure_keeps_local_work_possible(self) -> None:
        class FailingTask:
            @classmethod
            def init(cls, **kwargs: object) -> FakeTaskInstance:
                raise RuntimeError("credentials unavailable")

        tracker = ClearMLTracker(
            run_id="fallback-run",
            experiment_config=experiment_tracking(),
            project_config=project_tracking(),
            task_class=FailingTask,
        )
        with self.assertWarnsRegex(RuntimeWarning, "credentials unavailable"):
            tracker.start()
        tracker.log_metrics({"log_loss": 1.0}, namespace="validation", step=None)
        tracker.close()
        self.assertEqual(tracker.state["status"], "unavailable")
        self.assertIn("credentials unavailable", tracker.state["error"])


if __name__ == "__main__":
    unittest.main()
