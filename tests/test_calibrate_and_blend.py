from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/calibrate_and_blend.py"
SPEC = importlib.util.spec_from_file_location("calibrate_and_blend", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CalibrateAndBlendTest(unittest.TestCase):
    def test_candidate_weights_are_positive_convex_and_bounded(self) -> None:
        pair = MODULE.candidate_weights(2, (0.25, 0.5, 0.75))
        triple = MODULE.candidate_weights(3, (0.25, 0.5, 0.75))

        self.assertEqual(pair, [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)])
        self.assertEqual(len(triple), 4)
        self.assertIn((1 / 3, 1 / 3, 1 / 3), triple)
        for weights in [*pair, *triple]:
            self.assertTrue(all(value > 0 for value in weights))
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_temperature_calibration_preserves_probability_contract(self) -> None:
        probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
        calibrated = MODULE.calibrate(probabilities, 1.5)

        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue((calibrated >= 0).all())
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        np.testing.assert_array_equal(calibrated.argmax(axis=1), [0, 1])

    def test_fit_temperature_returns_finite_bounded_value(self) -> None:
        probabilities = np.array(
            [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.90, 0.05, 0.05]]
        )
        targets = np.array([0, 1, 2])

        temperature, calibrated = MODULE.fit_temperature(
            targets, probabilities, (0.5, 3.0)
        )

        self.assertGreaterEqual(temperature, 0.5)
        self.assertLessEqual(temperature, 3.0)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)

    def test_prediction_loader_accepts_owner_y_true_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame(
                {
                    "id": [10, 11],
                    "y_true": [0, 2],
                    "winner_model_a": [0.7, 0.1],
                    "winner_model_b": [0.2, 0.2],
                    "winner_tie": [0.1, 0.7],
                }
            ).to_csv(path, index=False)

            loaded = MODULE.load_prediction_set(
                {
                    "name": "candidate",
                    "checkpoint": "sha256:test",
                    "fold7_predictions": str(path),
                },
                "fold7",
            )

        np.testing.assert_array_equal(loaded.targets, [0, 2])
        np.testing.assert_allclose(loaded.probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
