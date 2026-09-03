from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.truncation import balanced_head_tail_truncate


class BalancedHeadTailTruncationTests(unittest.TestCase):
    def test_default_budget_is_balanced_between_responses(self) -> None:
        prompt, response_a, response_b, stats = balanced_head_tail_truncate(
            range(100),
            range(100, 200),
            range(200, 300),
            max_length=14,
            special_tokens=4,
        )
        self.assertEqual((len(prompt), len(response_a), len(response_b)), (2, 4, 4))
        self.assertEqual(stats.content_budget, 10)
        self.assertEqual(stats.allocated_lengths, (2, 4, 4))
        self.assertEqual(stats.unused_content_budget, 0)

    def test_unused_budget_is_redistributed_from_short_segments(self) -> None:
        prompt, response_a, response_b, stats = balanced_head_tail_truncate(
            [7],
            [10, 11],
            range(100, 120),
            max_length=13,
            special_tokens=3,
        )
        self.assertEqual(prompt, [7])
        self.assertEqual(response_a, [10, 11])
        self.assertEqual(len(response_b), 7)
        self.assertEqual(stats.output_lengths, (1, 2, 7))
        self.assertEqual(stats.used_content_tokens + stats.special_tokens, 13)

    def test_short_prompt_budget_is_shared_fairly_by_responses(self) -> None:
        result = balanced_head_tail_truncate(
            [],
            range(100),
            range(100, 200),
            max_length=11,
            special_tokens=2,
        )
        _, response_a, response_b, stats = result
        self.assertEqual(sum(stats.output_lengths), 9)
        self.assertLessEqual(abs(len(response_a) - len(response_b)), 1)
        self.assertEqual(stats.unused_content_budget, 0)

    def test_head_and_tail_are_preserved(self) -> None:
        prompt, response_a, response_b, _ = balanced_head_tail_truncate(
            list(range(10)),
            list(range(100, 110)),
            list(range(200, 210)),
            max_length=10,
            special_tokens=0,
        )
        self.assertEqual(prompt, [0, 9])
        self.assertEqual(response_a, [100, 101, 108, 109])
        self.assertEqual(response_b, [200, 201, 208, 209])

    def test_odd_keep_count_gives_extra_token_to_head(self) -> None:
        prompt, response_a, response_b, stats = balanced_head_tail_truncate(
            range(10),
            [],
            [],
            max_length=3,
            special_tokens=0,
        )
        self.assertEqual(prompt, [0, 1, 9])
        self.assertEqual(response_a, [])
        self.assertEqual(response_b, [])
        self.assertEqual(stats.output_lengths, (3, 0, 0))

    def test_sequences_are_copied_when_everything_fits(self) -> None:
        original_prompt = [1, 2]
        original_a = [3]
        original_b = [4, 5]
        prompt, response_a, response_b, stats = balanced_head_tail_truncate(
            original_prompt,
            original_a,
            original_b,
            max_length=12,
            special_tokens=3,
        )
        self.assertEqual((prompt, response_a, response_b), ([1, 2], [3], [4, 5]))
        self.assertIsNot(prompt, original_prompt)
        self.assertEqual(stats.truncated, (False, False, False))
        self.assertEqual(stats.unused_content_budget, 4)

    def test_is_deterministic_and_does_not_mutate_inputs(self) -> None:
        inputs = ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])
        snapshots = tuple(item.copy() for item in inputs)
        first = balanced_head_tail_truncate(
            *inputs, max_length=8, special_tokens=1
        )
        second = balanced_head_tail_truncate(
            *inputs, max_length=8, special_tokens=1
        )
        self.assertEqual(first, second)
        self.assertEqual(inputs, snapshots)

    def test_zero_content_budget_is_valid(self) -> None:
        prompt, response_a, response_b, stats = balanced_head_tail_truncate(
            [1], [2], [3], max_length=4, special_tokens=4
        )
        self.assertEqual((prompt, response_a, response_b), ([], [], []))
        self.assertEqual(stats.content_budget, 0)
        self.assertEqual(stats.truncated, (True, True, True))

    def test_budget_and_special_token_validation_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_length must be positive"):
            balanced_head_tail_truncate([], [], [], max_length=0, special_tokens=0)
        with self.assertRaisesRegex(ValueError, "cannot exceed max_length"):
            balanced_head_tail_truncate([], [], [], max_length=4, special_tokens=5)
        with self.assertRaises(TypeError):
            balanced_head_tail_truncate([], [], [], max_length=True, special_tokens=0)
        with self.assertRaises(ValueError):
            balanced_head_tail_truncate([], [], [], max_length=4, special_tokens=-1)

    def test_unfair_or_invalid_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be equal"):
            balanced_head_tail_truncate(
                [1], [2], [3], max_length=3, special_tokens=0,
                budget_weights=(1, 2, 3),
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            balanced_head_tail_truncate(
                [1], [2], [3], max_length=3, special_tokens=0,
                budget_weights=(0, 1, 1),
            )

    def test_invalid_token_ids_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            balanced_head_tail_truncate(
                [1, "2"], [3], [4], max_length=4, special_tokens=0
            )
        with self.assertRaises(ValueError):
            balanced_head_tail_truncate(
                [1, -2], [3], [4], max_length=4, special_tokens=0
            )
        with self.assertRaises(TypeError):
            balanced_head_tail_truncate(
                "123", [3], [4], max_length=4, special_tokens=0
            )


if __name__ == "__main__":
    unittest.main()
