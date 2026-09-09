from __future__ import annotations

import csv
import json
import re
import unittest

import numpy as np

from pmldl_llm.notebook import (
    ExperimentOutput,
    NotebookExperimentSetup,
    ask_experiment_setup,
    load_experiment_setup,
    prepare_experiment_config,
    run_notebook_experiment,
)

from test_experiment_run import ExperimentRunFixture


def notebook_setup(
    experiment_id: str = "E040", *, smoke_test: bool = True
) -> NotebookExperimentSetup:
    return NotebookExperimentSetup(
        experiment_id=experiment_id,
        owner="Danil",
        title="Notebook experiment",
        hypothesis="This notebook experiment should improve validation quality.",
        changed_factor="notebook runner",
        model_name="test-model",
        model_revision="revision-1",
        smoke_test=smoke_test,
        training={"epochs": 1},
        tags=["notebook"],
        tracking_mode="disabled",
    )


class NotebookExperimentTests(ExperimentRunFixture, unittest.TestCase):
    def test_single_runner_call_creates_valid_run_artifact_and_leaderboard(self) -> None:
        source = self.root / "model.bin"
        source.write_bytes(b"model")

        def train(_run: object) -> ExperimentOutput:
            probabilities = np.array(
                [
                    [0.80, 0.10, 0.10],
                    [0.10, 0.80, 0.10],
                    [0.10, 0.10, 0.80],
                ]
            )
            return ExperimentOutput.from_predictions(
                y_true=np.array([0, 1, 2]),
                original_probabilities=probabilities,
                swapped_back_probabilities=probabilities,
                artifacts={"models/model.bin": source},
            )

        result = run_notebook_experiment(
            train,
            notebook_setup(smoke_test=False),
            project_root=self.root,
        )

        self.assertTrue(result.result_dir.is_dir())
        self.assertEqual(
            (result.artifact_dir / "models/model.bin").read_bytes(), b"model"
        )
        self.assertTrue(result.leaderboard_updated)
        config = json.loads(result.config_path.read_text())
        self.assertEqual(config["experiment_id"], "E040")
        with (self.root / "results/leaderboard.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment_id"], "E040")
        self.assertAlmostEqual(float(rows[0]["log_loss"]), -np.log(0.8))
        self.assertEqual(rows[0]["swap_error_l1"], "0.0")

    def test_existing_experiment_id_cannot_be_silently_changed(self) -> None:
        setup = notebook_setup()
        prepare_experiment_config(setup, project_root=self.root)
        changed = notebook_setup()
        changed.title = "A different experiment"
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_experiment_config(changed, project_root=self.root)

    def test_invalid_experiment_id_is_rejected_before_writing(self) -> None:
        setup = notebook_setup("../escape")
        with self.assertRaisesRegex(ValueError, "experiment_id"):
            prepare_experiment_config(setup, project_root=self.root)
        self.assertFalse((self.root / "configs/escape.json").exists())

    def test_training_exception_is_recorded_as_failed_run(self) -> None:
        def train(_run: object) -> ExperimentOutput:
            raise RuntimeError("GPU failed")

        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            run_notebook_experiment(
                train,
                notebook_setup(),
                project_root=self.root,
            )
        run_directories = list((self.root / "results/runs").iterdir())
        self.assertEqual(len(run_directories), 1)
        metadata = json.loads((run_directories[0] / "run.json").read_text())
        self.assertEqual(metadata["status"], "failed")

    def test_prompt_collects_contract_fields(self) -> None:
        answers = iter(
            [
                "Prompted experiment",
                "Longer context should improve preference classification.",
                "maximum sequence length",
                "answerdotai/ModernBERT-base@revision-2",
            ]
        )
        setup = ask_experiment_setup(
            project_root=self.root, input_fn=lambda _prompt: next(answers)
        )
        self.assertRegex(setup.experiment_id, re.compile(r"^E[0-9]{20}$"))
        self.assertEqual(setup.seed, 42)
        self.assertTrue(setup.smoke_test)
        self.assertEqual(setup.model_name, "answerdotai/ModernBERT-base")
        self.assertEqual(setup.model_revision, "revision-2")
        self.assertEqual(setup.tags, ["self-service", "notebook"])
        self.assertEqual(setup.tracking_mode, "online")

    def test_load_experiment_setup_round_trips_versioned_config(self) -> None:
        setup = load_experiment_setup(
            "configs/experiments/E030.json", project_root=self.root
        )
        self.assertEqual(setup.experiment_id, "E030")
        self.assertEqual(setup.model_revision, "abc123")
        self.assertEqual(setup.training, {"epochs": 1})


if __name__ == "__main__":
    unittest.main()
