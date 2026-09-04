from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pmldl_llm.experiment import ExperimentRun


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def project_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "allowed_evaluation_roles": ["selection"],
        "primary_metric": {"name": "log_loss", "direction": "minimize"},
        "required_metrics": ["log_loss", "runtime_seconds"],
        "results": {
            "run_directory": "results/runs",
            "leaderboard": "results/leaderboard.csv",
            "metrics_schema": "configs/metrics.schema.json",
            "large_artifact_directory": "artifacts",
        },
        "tracking": {
            "project_name": "test-project",
            "required_tags": [
                "experiment_id",
                "owner",
                "track",
                "evaluation_role",
                "seed",
            ],
        },
        "run_policy": {
            "require_clean_git": False,
            "require_parent_for_controlled_experiment": True,
        },
    }


def experiment_config(*, smoke_test: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "E030",
        "owner": "Danil",
        "title": "Test experiment",
        "track": "neural",
        "status": "planned",
        "hypothesis": "A test hypothesis with enough detail.",
        "parent_experiment_id": None,
        "changed_factor": "test runner",
        "seed": 42,
        "evaluation_role": "selection",
        "smoke_test": smoke_test,
        "model": {"name": "test-model", "revision": "abc123"},
        "training": {"epochs": 1},
        "tracking": {"backend": "none", "mode": "disabled"},
        "tags": ["unit-test"],
    }


class ExperimentRunFixture:
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        write_json(self.root / "configs/project.json", project_config())
        write_json(
            self.root / "configs/experiments/E030.json", experiment_config()
        )


class ExperimentRunTests(ExperimentRunFixture, unittest.TestCase):
    def test_completed_run_persists_metrics_metadata_and_artifact(self) -> None:
        with ExperimentRun(
            "configs/experiments/E030.json", project_root=self.root
        ) as run:
            run.log_metrics({"log_loss": 1.02, "accuracy": 0.48})
            source = self.root / "temporary-model.bin"
            source.write_bytes(b"model")
            saved = run.log_artifact("models/model.bin", source)

        metadata = json.loads((run.result_dir / "run.json").read_text())
        metrics = json.loads((run.result_dir / "metrics.json").read_text())
        snapshot = json.loads((run.result_dir / "config.json").read_text())
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["tracking"]["status"], "disabled")
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["summary"]["validation"]["log_loss"], 1.02)
        self.assertGreaterEqual(
            metrics["summary"]["validation"]["runtime_seconds"], 0.0
        )
        self.assertEqual(snapshot["experiment_id"], "E030")
        self.assertEqual(saved.read_bytes(), b"model")
        self.assertTrue(str(run.result_dir).startswith(str(self.root / "results")))
        self.assertTrue(str(saved).startswith(str(self.root / "artifacts")))

    def test_failed_run_is_persisted_and_original_error_propagates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training exploded"):
            with ExperimentRun(
                "configs/experiments/E030.json", project_root=self.root
            ) as run:
                raise RuntimeError("training exploded")

        metadata = json.loads((run.result_dir / "run.json").read_text())
        metrics = json.loads((run.result_dir / "metrics.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "RuntimeError")
        self.assertIn("training exploded", metadata["error"]["message"])
        self.assertEqual(metrics["status"], "failed")

    def test_project_policy_blocks_locked_evaluation_role(self) -> None:
        config = experiment_config()
        config["evaluation_role"] = "final_holdout"
        write_json(self.root / "configs/experiments/E031.json", config)
        with self.assertRaisesRegex(ValueError, "locked"):
            ExperimentRun(
                "configs/experiments/E031.json", project_root=self.root
            )

    def test_full_run_fails_when_required_metric_is_missing(self) -> None:
        config = experiment_config(smoke_test=False)
        write_json(self.root / "configs/experiments/E032.json", config)
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            with ExperimentRun(
                "configs/experiments/E032.json", project_root=self.root
            ):
                pass

    def test_artifact_path_cannot_escape_run_directory(self) -> None:
        with ExperimentRun(
            "configs/experiments/E030.json", project_root=self.root
        ) as run:
            with self.assertRaisesRegex(ValueError, "inside"):
                run.artifact_path("../outside.bin")


if __name__ == "__main__":
    unittest.main()
