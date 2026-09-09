from __future__ import annotations

import csv
import json
import unittest

from pmldl_llm.experiment import ExperimentRun
from pmldl_llm.result_collection import collect_result_rows, write_leaderboard

from test_experiment_run import ExperimentRunFixture, experiment_config, write_json


class ResultCollectionTests(ExperimentRunFixture, unittest.TestCase):
    def create_run(
        self,
        experiment_id: str,
        loss: float,
        *,
        smoke_test: bool = False,
    ) -> ExperimentRun:
        config = experiment_config(smoke_test=smoke_test)
        config["experiment_id"] = experiment_id
        config["title"] = f"Experiment {experiment_id}"
        config_path = self.root / f"configs/experiments/{experiment_id}.json"
        write_json(config_path, config)
        with ExperimentRun(config_path, project_root=self.root) as run:
            run.log_metric("log_loss", loss)
        return run

    def test_collection_excludes_smoke_and_sorts_by_primary_metric(self) -> None:
        self.create_run("E030", 1.04)
        self.create_run("E031", 0.99)
        self.create_run("E032", 0.50, smoke_test=True)

        collection = collect_result_rows(project_root=self.root)
        self.assertEqual(
            [row["experiment_id"] for row in collection.rows],
            ["E031", "E030"],
        )
        self.assertEqual([row["rank"] for row in collection.rows], [1, 2])
        self.assertEqual(collection.excluded_counts["smoke_test"], 1)
        self.assertEqual(collection.invalid_runs, [])

        output = self.root / "results/leaderboard.csv"
        write_leaderboard(output, collection)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["experiment_id"] for row in rows], ["E031", "E030"])
        self.assertEqual(rows[0]["log_loss"], "0.99")
        self.assertIn("runtime_seconds", rows[0])

    def test_invalid_run_is_reported_and_not_collected(self) -> None:
        run = self.create_run("E030", 1.0)
        metrics_path = run.result_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["unexpected"] = 1
        metrics_path.write_text(json.dumps(metrics))

        collection = collect_result_rows(project_root=self.root)
        self.assertEqual(collection.rows, [])
        self.assertEqual(len(collection.invalid_runs), 1)
        self.assertTrue(
            any(
                "unknown fields" in error
                for error in collection.invalid_runs[0].errors
            )
        )

    def test_include_smoke_adds_completed_smoke_run(self) -> None:
        self.create_run("E030", 1.0, smoke_test=True)
        collection = collect_result_rows(
            project_root=self.root, include_smoke=True
        )
        self.assertEqual(len(collection.rows), 1)
        self.assertTrue(collection.rows[0]["smoke_test"])


if __name__ == "__main__":
    unittest.main()
