from __future__ import annotations

import json
import unittest

from pmldl_llm.experiment import ExperimentRun
from pmldl_llm.run_validation import validate_run_directory

from test_experiment_run import ExperimentRunFixture


class RunValidationTests(ExperimentRunFixture, unittest.TestCase):
    def create_completed_run(self) -> ExperimentRun:
        with ExperimentRun(
            "configs/experiments/E030.json", project_root=self.root
        ) as run:
            run.log_metric("log_loss", 1.02)
        return run

    def test_generated_run_satisfies_cross_file_contract(self) -> None:
        run = self.create_completed_run()
        report = validate_run_directory(
            run.result_dir,
            project_root=self.root,
            require_artifacts=True,
        )
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.errors, [])

    def test_identity_mismatch_is_rejected(self) -> None:
        run = self.create_completed_run()
        metrics_path = run.result_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["experiment_id"] = "E999"
        metrics_path.write_text(json.dumps(metrics))
        report = validate_run_directory(run.result_dir, project_root=self.root)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("experiment_id must match" in error for error in report.errors)
        )

    def test_non_finite_metric_is_rejected(self) -> None:
        run = self.create_completed_run()
        metrics_path = run.result_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["summary"]["validation"]["log_loss"] = float("nan")
        metrics_path.write_text(json.dumps(metrics))
        report = validate_run_directory(run.result_dir, project_root=self.root)
        self.assertFalse(report.valid)
        self.assertTrue(any("finite number" in error for error in report.errors))

    def test_artifacts_are_optional_unless_explicitly_required(self) -> None:
        run = self.create_completed_run()
        run.artifact_dir.rmdir()
        relaxed = validate_run_directory(run.result_dir, project_root=self.root)
        strict = validate_run_directory(
            run.result_dir,
            project_root=self.root,
            require_artifacts=True,
        )
        self.assertTrue(relaxed.valid, relaxed.errors)
        self.assertTrue(relaxed.warnings)
        self.assertFalse(strict.valid)


if __name__ == "__main__":
    unittest.main()
