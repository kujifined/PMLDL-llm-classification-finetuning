from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_release.py"
CLASS_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def prediction_frame(ids: list[int], targets: list[int] | None, rows: list[list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids, **dict(zip(CLASS_COLUMNS, np.asarray(rows).T))})
    if targets is not None:
        frame.insert(1, "target", targets)
    return frame


class FinalizeReleaseTests(unittest.TestCase):
    def test_frozen_ensemble_writes_holdout_metrics_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "final_model.json"
            config_path.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "models": [
                            {"name": "deberta_qlora_seed42", "weight": 0.5},
                            {"name": "sparse_tfidf_sprint2", "weight": 0.5},
                        ],
                        "calibration": {"temperature": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            deberta_fold9 = root / "deberta_fold9.csv"
            sparse_fold9 = root / "sparse_fold9.csv"
            deberta_test = root / "deberta_test.csv"
            sparse_test = root / "sparse_test.csv"
            prediction_frame(
                [1, 2, 3], [0, 1, 2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
            ).to_csv(deberta_fold9, index=False)
            prediction_frame(
                [3, 1, 2], [2, 0, 1], [[0.2, 0.1, 0.7], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
            ).to_csv(sparse_fold9, index=False)
            prediction_frame(
                [10, 11], None, [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
            ).to_csv(deberta_test, index=False)
            prediction_frame(
                [11, 10], None, [[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]]
            ).to_csv(sparse_test, index=False)

            output_dir = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(config_path),
                    "--fold9-predictions",
                    str(deberta_fold9),
                    str(sparse_fold9),
                    "--test-predictions",
                    str(deberta_test),
                    str(sparse_test),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            metrics = json.loads((output_dir / "final_holdout_metrics.json").read_text())
            self.assertEqual(metrics["evaluation_role"], "final_holdout")
            self.assertEqual(metrics["fold"], 9)
            self.assertLess(metrics["metrics"]["log_loss"], 0.5)
            submission = pd.read_csv(output_dir / "submission.csv")
            self.assertEqual(list(submission.columns), ["id", *CLASS_COLUMNS])
            np.testing.assert_allclose(submission.loc[:, CLASS_COLUMNS].sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
