from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from pmldl_llm.experiment import ExperimentRun
from pmldl_llm.notebook import NotebookExperimentSetup
from pmldl_llm.team_workflow import (
    _stage_and_commit,
    prepare_full_run,
    scaffold_experiment,
    slugify,
    submit_experiment,
)

from test_experiment_run import ExperimentRunFixture


class TeamWorkflowTests(ExperimentRunFixture, unittest.TestCase):
    def copy_template(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        template_source = repository_root / "output/jupyter-notebook/team-managed-experiment.ipynb"
        template_target = (
            self.root / "output/jupyter-notebook/team-managed-experiment.ipynb"
        )
        template_target.parent.mkdir(parents=True)
        shutil.copy2(template_source, template_target)
        shutil.copy2(repository_root / ".gitignore", self.root / ".gitignore")

    def test_scaffold_creates_config_and_bound_notebook(self) -> None:
        self.copy_template()
        setup = NotebookExperimentSetup(
            experiment_id="E20260905000000000000",
            owner="Karim",
            title="Long context test",
            hypothesis="Longer context should improve validation log loss.",
            changed_factor="maximum sequence length",
            model_name="example/model",
            smoke_test=True,
            tracking_mode="disabled",
        )

        result = scaffold_experiment(
            root=self.root,
            setup=setup,
            create_branch=False,
        )

        config = json.loads(result.config_path.read_text(encoding="utf-8"))
        notebook_text = result.notebook_path.read_text(encoding="utf-8")
        self.assertEqual(
            config["training"]["notebook"],
            result.notebook_path.relative_to(self.root).as_posix(),
        )
        self.assertIn("configs/experiments/E20260905000000000000.json", notebook_text)
        self.assertNotIn("__EXPERIMENT_CONFIG__", notebook_text)
        self.assertEqual(result.branch, "experiment/E20260905000000000000-long-context-test")

    def test_slugify_has_a_safe_fallback(self) -> None:
        self.assertEqual(slugify("ModernBERT + LoRA"), "modernbert-lora")
        self.assertEqual(slugify("Эксперимент"), "experiment")

    def test_prepare_and_submit_create_clean_commits(self) -> None:
        self.copy_template()
        (self.root / "tests").mkdir()
        (self.root / "tests/test_smoke.py").write_text(
            "import unittest\n\n"
            "class SmokeTest(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=self.root, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"], cwd=self.root, check=True
        )
        setup = NotebookExperimentSetup(
            experiment_id="E20260905000000000001",
            owner="Test User",
            title="Automated branch",
            hypothesis="The automated workflow should preserve a clean checkout.",
            changed_factor="workflow automation",
            model_name="example/model",
            smoke_test=True,
            tracking_mode="disabled",
        )
        scaffold = scaffold_experiment(root=self.root, setup=setup)
        with ExperimentRun(scaffold.config_path, project_root=self.root) as run:
            run.log_metric("log_loss", 1.0)

        commit = prepare_full_run(self.root, setup.experiment_id)

        config = json.loads(scaffold.config_path.read_text(encoding="utf-8"))
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        self.assertFalse(config["smoke_test"])
        self.assertEqual(status, "")
        self.assertEqual(len(commit), 40)

        with ExperimentRun(scaffold.config_path, project_root=self.root) as full_run:
            full_run.log_metric("log_loss", 0.9)
        result_commit, compare_url = submit_experiment(
            self.root,
            setup.experiment_id,
            push=False,
        )
        leaderboard = (self.root / "results/leaderboard.csv").read_text(
            encoding="utf-8"
        )
        final_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        self.assertIn(setup.experiment_id, leaderboard)
        self.assertIsNone(compare_url)
        self.assertEqual(final_status, "")
        self.assertEqual(len(result_commit), 40)

    def test_scoped_commit_leaves_unrelated_untracked_credentials_alone(self) -> None:
        (self.root / "allowed.txt").write_text("safe\n", encoding="utf-8")
        (self.root / "clearml.conf").write_text("secret\n", encoding="utf-8")
        subprocess.run(["git", "init", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=self.root, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.root,
            check=True,
        )

        commit = _stage_and_commit(
            self.root,
            "Scoped commit",
            self.root / "allowed.txt",
        )

        self.assertEqual(len(commit), 40)
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        self.assertEqual(tracked, "allowed.txt\n")

    def test_prepare_full_restores_files_when_tests_fail(self) -> None:
        self.copy_template()
        (self.root / "tests").mkdir()
        (self.root / "tests/test_failure.py").write_text(
            "import unittest\n\n"
            "class FailureTest(unittest.TestCase):\n"
            "    def test_failure(self):\n"
            "        self.fail('expected')\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-b", "main"], cwd=self.root, check=True)
        setup = NotebookExperimentSetup(
            experiment_id="E20260905000000000002",
            owner="Test User",
            title="Rollback failed transition",
            hypothesis="A failed gate must leave the smoke contract unchanged.",
            changed_factor="transactional prepare-full",
            model_name="example/model",
            smoke_test=True,
            tracking_mode="disabled",
        )
        scaffold = scaffold_experiment(
            root=self.root,
            setup=setup,
            create_branch=False,
        )
        with ExperimentRun(scaffold.config_path, project_root=self.root) as run:
            run.log_metric("log_loss", 1.0)
        config_before = scaffold.config_path.read_bytes()
        notebook_before = scaffold.notebook_path.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "Tests failed"):
            prepare_full_run(self.root, setup.experiment_id)

        self.assertEqual(scaffold.config_path.read_bytes(), config_before)
        self.assertEqual(scaffold.notebook_path.read_bytes(), notebook_before)

    def test_scoped_commit_rejects_unexpected_pre_staged_file(self) -> None:
        (self.root / "allowed.txt").write_text("safe\n", encoding="utf-8")
        (self.root / "unexpected.txt").write_text("other\n", encoding="utf-8")
        subprocess.run(["git", "init", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=self.root, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "add", "unexpected.txt"], cwd=self.root, check=True
        )

        with self.assertRaisesRegex(RuntimeError, "staged outside this experiment"):
            _stage_and_commit(
                self.root,
                "Scoped commit",
                self.root / "allowed.txt",
            )


if __name__ == "__main__":
    unittest.main()
