from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_bias_baseline_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_bias_baseline_experiment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BiasBaselineExperimentTests(unittest.TestCase):
    def test_canonical_metrics_add_brier_and_rename_swap_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "validation_predictions.csv"
            pd.DataFrame(
                {
                    "target": [0, 1],
                    "winner_model_a": [0.8, 0.1],
                    "winner_model_b": [0.1, 0.7],
                    "winner_tie": [0.1, 0.2],
                }
            ).to_csv(predictions, index=False)
            evaluation = {
                "bias_logistic": {
                    "log_loss": 0.2,
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "ece_15": 0.15,
                },
                "raw_symmetry_l1": 0.01,
            }

            metrics = MODULE.canonical_metrics(evaluation, predictions)

        self.assertEqual(
            set(metrics),
            {
                "log_loss",
                "accuracy",
                "macro_f1",
                "ece_15",
                "brier_score",
                "swap_error_l1",
            },
        )
        self.assertAlmostEqual(metrics["brier_score"], 0.1)
        self.assertEqual(metrics["swap_error_l1"], 0.01)


if __name__ == "__main__":
    unittest.main()
