from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.constants import TARGET_COLUMNS
from pmldl_llm.config import validate_fold_roles
from pmldl_llm.data import (
    DataContractError,
    clean_turns,
    file_sha256,
    swap_probability_columns,
    swap_target_indices,
    swapped_frame,
    validate_train_frame,
    verify_competition_data_dir,
    verify_checksum_manifest,
)
from pmldl_llm.features import build_bias_features
from pmldl_llm.split import load_frozen_folds, make_group_folds
from pmldl_llm.submission import build_submission, write_submission
from pmldl_llm.text import flatten_conversation


def tiny_training_frame() -> pd.DataFrame:
    rows = []
    row_id = 0
    for group in range(12):
        label = group % 3
        for duplicate in range(2):
            targets = [0, 0, 0]
            targets[label] = 1
            rows.append(
                {
                    "id": row_id,
                    "model_a": f"a_{group}",
                    "model_b": f"b_{group}",
                    "prompt": f'["shared prompt {group}"]',
                    "response_a": '["answer A"]',
                    "response_b": '["answer B"]',
                    TARGET_COLUMNS[0]: targets[0],
                    TARGET_COLUMNS[1]: targets[1],
                    TARGET_COLUMNS[2]: targets[2],
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


class CoreContractTests(unittest.TestCase):
    def test_json_null_is_cleaned(self) -> None:
        self.assertEqual(clean_turns('["ok", null]'), ["ok", ""])
        self.assertIn("null_response", flatten_conversation('["ok", null]'))

    def test_swap_is_an_involution(self) -> None:
        frame = tiny_training_frame().iloc[:3]
        pd.testing.assert_frame_equal(
            swapped_frame(swapped_frame(frame)),
            frame,
        )
        targets = np.array([0, 1, 2])
        np.testing.assert_array_equal(
            swap_target_indices(swap_target_indices(targets)),
            targets,
        )
        probabilities = np.array([[0.6, 0.3, 0.1]])
        np.testing.assert_allclose(
            swap_probability_columns(swap_probability_columns(probabilities)),
            probabilities,
        )

    def test_group_split_has_no_prompt_leakage(self) -> None:
        frame = tiny_training_frame()
        validate_train_frame(frame)
        folds = make_group_folds(frame, n_splits=3, seed=7)
        self.assertEqual(folds.groupby("prompt_group")["fold"].nunique().max(), 1)
        self.assertEqual(set(folds["fold"]), {0, 1, 2})

    def test_frozen_split_round_trip_and_tamper_detection(self) -> None:
        frame = tiny_training_frame()
        folds = make_group_folds(frame, n_splits=3, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "folds.csv"
            config_path = root / "split.json"
            metadata_path = root / "metadata.json"
            config_path.write_text('{"test": true}\n', encoding="utf-8")
            folds.to_csv(path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "folds_sha256": file_sha256(path),
                        "split_config_sha256": file_sha256(config_path),
                        "dataset_sha256": {},
                        "rows": len(folds),
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_frozen_folds(
                frame,
                path,
                config_path,
                n_splits=3,
                dataset_hashes={},
            )
            pd.testing.assert_frame_equal(loaded, folds)
            tampered = folds.copy()
            tampered.loc[0, "prompt_group"] = "0" * 20
            tampered.to_csv(path, index=False)
            with self.assertRaises(DataContractError):
                load_frozen_folds(
                    frame,
                    path,
                    config_path,
                    n_splits=3,
                    dataset_hashes={},
                )

    def test_fold_role_contract_is_exhaustive_and_disjoint(self) -> None:
        config = {
            "n_splits": 5,
            "seed": 7,
            "training_folds": [0, 1],
            "validation_fold": 2,
            "calibration_fold": 3,
            "final_holdout_fold": 4,
        }
        roles = validate_fold_roles(config)
        self.assertEqual(roles.training_folds, (0, 1))
        invalid = {**config, "calibration_fold": 2}
        with self.assertRaises(ValueError):
            validate_fold_roles(invalid)

    def test_checksum_manifest_detects_changed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("verified", encoding="utf-8")
            manifest = root / "checksums.sha256"
            manifest.write_text(
                f"{file_sha256(payload)}  payload.txt\n", encoding="utf-8"
            )
            self.assertEqual(
                verify_checksum_manifest(manifest)["payload.txt"],
                file_sha256(payload),
            )
            payload.write_text("changed", encoding="utf-8")
            with self.assertRaises(DataContractError):
                verify_checksum_manifest(manifest)

    def test_checksum_manifest_is_bound_to_loaded_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_hashes = {}
            for filename in ("train.csv", "test.csv", "sample_submission.csv"):
                path = root / filename
                path.write_text(filename, encoding="utf-8")
                expected_hashes[f"snapshot/{filename}"] = file_sha256(path)
            verified = verify_competition_data_dir(root, expected_hashes)
            self.assertEqual(verified, expected_hashes)
            (root / "test.csv").write_text("different", encoding="utf-8")
            with self.assertRaises(DataContractError):
                verify_competition_data_dir(root, expected_hashes)

    def test_feature_schema_is_stable_under_swap(self) -> None:
        frame = tiny_training_frame().iloc[:2]
        original, original_names = build_bias_features(frame)
        swapped, swapped_names = build_bias_features(swapped_frame(frame))
        self.assertEqual(original_names, swapped_names)
        self.assertEqual(original.shape, swapped.shape)

    def test_submission_contract(self) -> None:
        ids = np.array([10, 20])
        probabilities = np.array([[0.2, 0.3, 0.5], [2.0, 1.0, 1.0]])
        submission = build_submission(ids, probabilities)
        self.assertEqual(list(submission.columns), ["id", *TARGET_COLUMNS])
        np.testing.assert_allclose(
            submission[list(TARGET_COLUMNS)].sum(axis=1),
            1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_submission(
                ids, probabilities, Path(directory) / "submission.csv"
            )
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
