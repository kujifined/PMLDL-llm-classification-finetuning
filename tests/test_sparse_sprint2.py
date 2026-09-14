from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sweep_sparse_sprint2 import (  # noqa: E402
    Candidate,
    candidates_for_stage,
    changed_candidate_fields,
    control_from_parent,
    fit_candidate,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(18):
        rows.append(
            {
                "id": index,
                "model_a": f"a-{index}",
                "model_b": f"b-{index}",
                "prompt": f'["Explain token {index % 3}"]',
                "response_a": f'["Concise answer {index}!"]',
                "response_b": f'["Verbose response {index} with detail?"]',
                "winner_model_a": int(index % 3 == 0),
                "winner_model_b": int(index % 3 == 1),
                "winner_tie": int(index % 3 == 2),
            }
        )
    return pd.DataFrame(rows)


class SparseSprint2Tests(unittest.TestCase):
    def test_control_is_loaded_exactly_from_verified_b2(self) -> None:
        parent = json.loads(
            (PROJECT_ROOT / "configs/experiments/E20260909125010457517.json")
            .read_text(encoding="utf-8")
        )
        control = control_from_parent(parent, alpha=0.0003)

        self.assertEqual(control.word_ngram_range, (1, 2))
        self.assertEqual(control.character_ngram_range, (3, 5))
        self.assertEqual(control.word_max_features, 75000)
        self.assertEqual(control.character_max_features, 50000)
        self.assertEqual(control.word_min_df, 3)
        self.assertEqual(control.character_min_df, 5)
        self.assertEqual(control.alpha, 0.0003)

    def test_each_sweep_candidate_changes_exactly_one_field(self) -> None:
        control = Candidate(
            "control",
            "control",
            (1, 2),
            (3, 5),
            75000,
            50000,
            3,
            5,
            0.995,
            0.0003,
        )
        sweep = {
            "word_ngram_range": [[1, 1], [1, 3]],
            "character_ngram_range": [[2, 5], [3, 6]],
            "word_max_features": [50000, 100000],
            "character_max_features": [25000, 75000],
            "word_min_df": [1, 5],
            "character_min_df": [3, 8],
            "alpha": [0.001, 0.0001],
        }
        for stage in sweep:
            with self.subTest(stage=stage):
                candidates = candidates_for_stage(control, stage, sweep)
                self.assertEqual(len(candidates), 2)
                for candidate in candidates:
                    self.assertEqual(changed_candidate_fields(control, candidate), {stage})

    def test_fit_candidate_runs_word_and_char_b2_pipeline(self) -> None:
        frame = sample_frame()
        training = frame.iloc[:12].reset_index(drop=True)
        validation = frame.iloc[12:].reset_index(drop=True)
        training_targets = np.arange(12) % 3
        validation_targets = np.arange(12, 18) % 3
        candidate = Candidate(
            "tiny_control",
            "control",
            (1, 2),
            (3, 5),
            200,
            300,
            1,
            1,
            1.0,
            0.0003,
        )

        result = fit_candidate(
            candidate,
            training,
            validation,
            training_targets,
            validation_targets,
            seed=42,
            max_iter=2,
            average=True,
            keep_objects=True,
        )

        self.assertEqual(result["candidate"], candidate)
        expected_columns = (
            2 * result["metrics"]["word_vocabulary_size"]
            + 2 * result["metrics"]["character_vocabulary_size"]
            + 64
        )
        self.assertEqual(result["metrics"]["feature_columns"], expected_columns)
        self.assertTrue(np.isfinite(result["metrics"]["log_loss"]))
        self.assertEqual(
            result["original_validation_probabilities"].shape,
            (len(validation), 3),
        )


if __name__ == "__main__":
    unittest.main()
