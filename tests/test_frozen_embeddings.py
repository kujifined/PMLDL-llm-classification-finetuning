from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pmldl_llm.frozen_embeddings import EmbeddingCache


class CountingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, values: list[str], **_kwargs: object) -> np.ndarray:
        self.calls.append(list(values))
        return np.asarray(
            [[float(len(value)), float(index + 1)] for index, value in enumerate(values)],
            dtype=np.float32,
        )


class FrozenEmbeddingCacheTests(unittest.TestCase):
    def test_reuses_rows_and_keeps_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(
                root=Path(directory),
                model_name="test-model",
                model_revision="test-revision",
                train_sha256="train-hash",
                test_sha256="test-hash",
                max_seq_length=384,
            )
            encoder = CountingEncoder()
            first = cache.get_or_compute(
                split="train",
                ids=["row-a", "row-b"],
                text_parts=(
                    ["prompt a", "prompt b"],
                    ["answer a", "answer b"],
                    ["answer b", "answer a"],
                ),
                encoder=encoder,
                batch_size=2,
                show_progress_bar=False,
            )
            second = cache.get_or_compute(
                split="train",
                ids=["row-b", "row-a", "row-c"],
                text_parts=(
                    ["prompt b", "prompt a", "prompt c"],
                    ["answer b", "answer a", "answer c"],
                    ["answer a", "answer b", "answer c"],
                ),
                encoder=encoder,
                batch_size=2,
                show_progress_bar=False,
            )

            self.assertEqual(first[1]["computed_rows"], 2)
            self.assertEqual(second[1]["computed_rows"], 1)
            self.assertEqual(second[1]["cached_rows"], 2)
            self.assertEqual(len(encoder.calls), 6)
            np.testing.assert_allclose(second[0][:2], first[0][[1, 0]])
            self.assertTrue((cache.directory / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
