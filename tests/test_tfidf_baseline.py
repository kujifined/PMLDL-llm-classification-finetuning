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
from pmldl_llm.data import (
    file_sha256,
    prompt_group_ids,
    swap_target_indices,
    swapped_frame,
    target_indices,
)
from pmldl_llm.notebook import NotebookExperimentSetup
from pmldl_llm.tfidf_baseline import (
    TfidfChannelConfig,
    WordCharacterFeatureEncoder,
    _stratified_sample,
    run_tfidf_experiment,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "id": index,
                "model_a": f"a-{index}",
                "model_b": f"b-{index}",
                "prompt": f'["Explain token {index % 3}", "Follow up {index}"]',
                "response_a": f'["Concise answer {index}!", "detail"]',
                "response_b": f'["Verbose response {index}?", "more detail"]',
                "winner_model_a": int(index % 3 == 0),
                "winner_model_b": int(index % 3 == 1),
                "winner_tie": int(index % 3 == 2),
            }
        )
    return pd.DataFrame(rows)


class FakeRun:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.logged_metrics: dict[str, dict[str, float]] = {}

    def artifact_path(self, relative_path: str) -> Path:
        path = self.artifact_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        namespace: str,
    ) -> None:
        self.logged_metrics[namespace] = metrics


class WordCharacterTfidfTests(unittest.TestCase):
    def test_encoder_combines_word_character_and_dense_features(self) -> None:
        frame = sample_frame()
        word = TfidfChannelConfig("word", (1, 2), 200, 1, 1.0)
        character = TfidfChannelConfig("char_wb", (3, 5), 300, 1, 1.0)
        encoder, original, swapped = WordCharacterFeatureEncoder.fit(
            frame,
            word,
            character,
        )

        self.assertGreater(len(encoder.word_vectorizer.vocabulary_), 0)
        self.assertGreater(len(encoder.character_vectorizer.vocabulary_), 0)
        self.assertEqual(original.shape, swapped.shape)
        self.assertEqual(original.shape[0], len(frame))
        self.assertGreater(
            original.shape[1],
            2 * len(encoder.word_vectorizer.vocabulary_),
        )
        self.assertTrue(np.isfinite(original.data).all())
        self.assertTrue(np.isfinite(swapped.data).all())

        transformed, transformed_swapped = encoder.transform(frame)
        np.testing.assert_allclose(transformed.toarray(), original.toarray())
        np.testing.assert_allclose(transformed_swapped.toarray(), swapped.toarray())

    def test_swapped_frame_produces_reversed_order_features(self) -> None:
        frame = sample_frame()
        word = TfidfChannelConfig("word", (1, 2), 200, 1, 1.0)
        character = TfidfChannelConfig("char_wb", (3, 5), 300, 1, 1.0)
        encoder, _, swapped_features = WordCharacterFeatureEncoder.fit(
            frame,
            word,
            character,
        )

        reversed_original, _ = encoder.transform(swapped_frame(frame))
        np.testing.assert_allclose(
            reversed_original.toarray(),
            swapped_features.toarray(),
        )

    def test_smoke_sample_is_deterministic_and_keeps_every_class(self) -> None:
        frame = sample_frame()
        targets = np.arange(len(frame)) % 3
        sampled, sampled_targets = _stratified_sample(frame, targets, 6, seed=42)
        repeated, repeated_targets = _stratified_sample(frame, targets, 6, seed=42)

        pd.testing.assert_frame_equal(sampled, repeated)
        np.testing.assert_array_equal(sampled_targets, repeated_targets)
        self.assertEqual(set(sampled_targets), {0, 1, 2})
        np.testing.assert_array_equal(
            swap_target_indices(swap_target_indices(sampled_targets)),
            sampled_targets,
        )

    def test_experiment_runs_end_to_end_on_verified_tiny_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data" / "llm-classification-finetuning"
            split_dir = root / "data" / "splits"
            config_dir = root / "configs"
            data_dir.mkdir(parents=True)
            split_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)

            train = pd.concat(
                [sample_frame(), sample_frame().assign(id=lambda x: x.id + 12)]
            )
            train = train.reset_index(drop=True)
            test = train.loc[:2, ["id", "prompt", "response_a", "response_b"]]
            sample_submission = pd.DataFrame(
                {
                    "id": test["id"],
                    TARGET_COLUMNS[0]: 1 / 3,
                    TARGET_COLUMNS[1]: 1 / 3,
                    TARGET_COLUMNS[2]: 1 / 3,
                }
            )
            train.to_csv(data_dir / "train.csv", index=False)
            test.to_csv(data_dir / "test.csv", index=False)
            sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

            manifest_entries = {}
            for filename in ("train.csv", "test.csv", "sample_submission.csv"):
                relative = f"llm-classification-finetuning/{filename}"
                manifest_entries[relative] = file_sha256(data_dir / filename)
            manifest = root / "data" / "checksums.sha256"
            manifest.write_text(
                "".join(
                    f"{digest}  {relative}\n"
                    for relative, digest in manifest_entries.items()
                ),
                encoding="utf-8",
            )

            split_config = {
                "seed": 42,
                "n_splits": 4,
                "training_folds": [0],
                "validation_fold": 1,
                "calibration_fold": 2,
                "final_holdout_fold": 3,
            }
            split_config_path = config_dir / "split.json"
            split_config_path.write_text(
                json.dumps(split_config),
                encoding="utf-8",
            )
            folds = pd.DataFrame(
                {
                    "id": train["id"],
                    "prompt_group": prompt_group_ids(train["prompt"]),
                    "fold": np.arange(len(train)) % 4,
                }
            )
            fold_path = split_dir / "folds.csv"
            folds.to_csv(fold_path, index=False)
            (split_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "folds_sha256": file_sha256(fold_path),
                        "split_config_sha256": file_sha256(split_config_path),
                        "dataset_sha256": manifest_entries,
                        "rows": len(folds),
                    }
                ),
                encoding="utf-8",
            )
            setup = NotebookExperimentSetup(
                experiment_id="E999",
                owner="tester",
                title="Word character TF-IDF",
                hypothesis="Word and character features improve log loss.",
                changed_factor="Add sparse text features.",
                model_name="word-char-tfidf",
                track="classical",
                parent_experiment_id="E002",
                seed=42,
                evaluation_role="selection",
                smoke_test=False,
                training={
                    "word_tfidf": {
                        "ngram_range": [1, 2],
                        "max_features": 200,
                        "min_df": 1,
                        "max_df": 1.0,
                    },
                    "character_tfidf": {
                        "analyzer": "char_wb",
                        "ngram_range": [3, 5],
                        "max_features": 300,
                        "min_df": 1,
                        "max_df": 1.0,
                    },
                    "classifier": {
                        "loss": "log_loss",
                        "alpha_candidates": [0.001, 0.0003, 0.0001],
                        "max_iter": 5,
                        "average": True,
                    },
                    "smoke_train_rows": 6,
                    "smoke_validation_rows": 3,
                },
            )
            run = FakeRun(root / "artifacts")

            output = run_tfidf_experiment(run, setup, root)

            self.assertEqual(
                set(output.metrics),
                {
                    "log_loss",
                    "accuracy",
                    "macro_f1",
                    "ece_15",
                    "brier_score",
                    "swap_error_l1",
                },
            )
            self.assertEqual(set(run.logged_metrics), {"diagnostics"})
            self.assertEqual(run.logged_metrics["diagnostics"]["training_rows"], 6)
            self.assertEqual(run.logged_metrics["diagnostics"]["validation_rows"], 6)
            self.assertTrue(
                all(Path(path).is_file() for path in output.artifacts.values())
            )


if __name__ == "__main__":
    unittest.main()
