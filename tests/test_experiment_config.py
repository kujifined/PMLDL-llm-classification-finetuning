from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from pmldl_llm.constants import TARGET_COLUMNS
from pmldl_llm.experiment import validate_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object in {path}")
    return value


class ExperimentConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = load_json(CONFIG_DIR / "project.json")
        self.schema = load_json(CONFIG_DIR / "experiment.schema.json")
        self.metrics_schema = load_json(CONFIG_DIR / "metrics.schema.json")
        self.template = load_json(CONFIG_DIR / "experiments" / "template.json")

    def test_template_contains_every_required_schema_field(self) -> None:
        required = set(self.schema["required"])
        properties = set(self.schema["properties"])
        self.assertEqual(required.difference(self.template), set())
        self.assertEqual(set(self.template).difference(properties), set())

    def test_template_obeys_shared_project_policy(self) -> None:
        self.assertEqual(self.template["schema_version"], self.project["schema_version"])
        self.assertRegex(str(self.template["experiment_id"]), re.compile(r"^E[0-9]{3,}$"))
        self.assertIn(
            self.template["evaluation_role"],
            self.project["allowed_evaluation_roles"],
        )
        self.assertEqual(self.template["seed"], self.project["default_seed"])
        self.assertEqual(self.template["tracking"]["backend"], self.project["tracking"]["backend"])
        self.assertEqual(
            self.template["tracking"]["mode"],
            self.project["tracking"]["default_mode"],
        )

    def test_every_versioned_experiment_config_is_valid(self) -> None:
        paths = sorted((CONFIG_DIR / "experiments").glob("*.json"))
        self.assertGreaterEqual(len(paths), 2)
        experiment_ids: set[str] = set()
        for path in paths:
            experiment = load_json(path)
            validate_experiment_config(experiment, self.project)
            if path.name == "template.json":
                continue
            experiment_id = str(experiment["experiment_id"])
            self.assertNotIn(experiment_id, experiment_ids, path.name)
            experiment_ids.add(experiment_id)

    def test_project_uses_the_competition_target_order(self) -> None:
        self.assertEqual(tuple(self.project["class_names"]), TARGET_COLUMNS)
        self.assertEqual(self.project["primary_metric"]["name"], "log_loss")
        self.assertEqual(self.project["primary_metric"]["direction"], "minimize")

    def test_shared_validation_files_exist(self) -> None:
        validation = self.project["validation"]
        for relative_path in validation.values():
            self.assertTrue((PROJECT_ROOT / relative_path).is_file(), relative_path)

    def test_metrics_schema_is_registered_and_has_canonical_fields(self) -> None:
        schema_path = PROJECT_ROOT / self.project["results"]["metrics_schema"]
        self.assertTrue(schema_path.is_file())
        self.assertEqual(
            set(self.metrics_schema["required"]),
            {
                "schema_version",
                "run_id",
                "experiment_id",
                "status",
                "evaluation_role",
                "seed",
                "primary_metric",
                "summary",
                "history",
            },
        )


if __name__ == "__main__":
    unittest.main()
