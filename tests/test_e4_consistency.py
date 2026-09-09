from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import numpy as np

from pmldl_llm.e4_consistency import (
    E2_EXPERIMENT_ID,
    consistency_estimator_weight,
    mean_js_divergence,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E20260909125807233308"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
NOTEBOOK_PATH = (
    ROOT
    / "output"
    / "jupyter-notebook"
    / f"{EXPERIMENT_ID}__e4-qlora-ab-consistency.ipynb"
)
LAUNCHER_PATH = ROOT / "output" / "kaggle" / "run_e4_consistency_experiment.ipynb"
EXPECTED_IMPLEMENTATION_COMMIT = "c3c443f8dbca95e8f792d435dead843006b285bb"


class E4ConsistencyTests(unittest.TestCase):
    def test_js_divergence_is_zero_for_identical_probabilities(self) -> None:
        probabilities = np.asarray(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64
        )
        self.assertAlmostEqual(
            mean_js_divergence(probabilities, probabilities), 0.0, places=12
        )

    def test_js_divergence_is_symmetric_and_positive(self) -> None:
        first = np.asarray([[0.8, 0.1, 0.1]], dtype=np.float64)
        second = np.asarray([[0.1, 0.8, 0.1]], dtype=np.float64)
        forward = mean_js_divergence(first, second)
        backward = mean_js_divergence(second, first)
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(forward, backward, places=12)

    def test_js_divergence_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            mean_js_divergence(np.ones((2, 3)), np.ones((1, 3)))

    def test_sparse_consistency_estimator_has_unit_mean_weight(self) -> None:
        stride = 16
        weights = [
            consistency_estimator_weight(index, stride=stride)
            for index in range(stride)
        ]
        self.assertEqual(weights[0], float(stride))
        self.assertEqual(sum(weight > 0 for weight in weights), 1)
        self.assertAlmostEqual(float(np.mean(weights)), 1.0)

    def test_sparse_consistency_estimator_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            consistency_estimator_weight(-1, stride=16)
        with self.assertRaisesRegex(ValueError, "positive"):
            consistency_estimator_weight(0, stride=0)

    def test_config_freezes_the_e2_control_and_single_changed_factor(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        training = config["training"]
        self.assertEqual(config["parent_experiment_id"], E2_EXPERIMENT_ID)
        self.assertIsInstance(config["smoke_test"], bool)
        self.assertEqual(config["evaluation_role"], "selection")
        self.assertEqual(training["epochs"], 3)
        self.assertEqual(training["effective_batch_size"], 32)
        self.assertEqual(training["max_length"], 512)
        self.assertEqual(training["consistency_lambda"], 0.1)
        self.assertEqual(
            training["consistency_estimator"],
            {
                "kind": "deterministic_stride",
                "microbatch_stride": 16,
                "importance_weighting": True,
            },
        )
        self.assertEqual(
            training["preprocessing"]["random_ab_swap_probability"], 0.5
        )
        self.assertEqual(training["control"]["experiment_id"], E2_EXPERIMENT_ID)

    def test_managed_notebook_is_clean_and_bound_to_e4(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("run_e4_consistency_experiment", source)
        self.assertIn("run_notebook_experiment", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                ast.parse("".join(cell.get("source", [])))

    def test_kaggle_launcher_is_reproducible_and_secret_safe(self) -> None:
        notebook = json.loads(LAUNCHER_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(EXPECTED_IMPLEMENTATION_COMMIT, source)
        self.assertIn("https://github.com/kujifined/", source)
        self.assertIn("UserSecretsClient", source)
        self.assertIn("CLEARML_API_ACCESS_KEY", source)
        self.assertIn("CLEARML_API_SECRET_KEY", source)
        self.assertNotIn("kaggle.json", source)
        self.assertIn("e4-run-output.zip", source)
        self.assertTrue(notebook["metadata"]["kaggle"]["isGpuEnabled"])
        self.assertTrue(notebook["metadata"]["kaggle"]["isInternetEnabled"])
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
