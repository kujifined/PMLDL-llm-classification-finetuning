from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "output"
    / "jupyter-notebook"
    / "team-managed-experiment.ipynb"
)


class NotebookTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_notebook_has_valid_top_level_structure(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIsInstance(self.notebook["cells"], list)
        self.assertGreaterEqual(len(self.notebook["cells"]), 8)
        self.assertEqual(self.notebook["cells"][0]["cell_type"], "markdown")

    def test_every_code_cell_is_clean_and_syntactically_valid(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]))

    def test_template_uses_the_managed_runner(self) -> None:
        source = "\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"]
        )
        self.assertIn("load_experiment_setup", source)
        self.assertIn("__EXPERIMENT_CONFIG__", source)
        self.assertIn("run_notebook_experiment", source)
        self.assertIn("ExperimentOutput", source)


if __name__ == "__main__":
    unittest.main()
