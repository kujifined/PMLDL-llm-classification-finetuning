"""Tokenizer-independent balanced truncation for prompt/response triples.

The caller is responsible for inserting special tokens after truncation.  The
``special_tokens`` argument reserves their total space in ``max_length``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence


SegmentLengths = tuple[int, int, int]


@dataclass(frozen=True)
class TruncationStats:
    """Accounting information for one truncation operation.

    Tuple-valued fields always use the order ``(prompt, response_a,
    response_b)``.
    """

    max_length: int
    special_tokens: int
    content_budget: int
    budget_weights: SegmentLengths
    original_lengths: SegmentLengths
    allocated_lengths: SegmentLengths
    output_lengths: SegmentLengths
    truncated: tuple[bool, bool, bool]
    used_content_tokens: int
    unused_content_budget: int


def _strict_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative, got {integer}.")
    return integer


def _validate_token_ids(token_ids: Iterable[int], *, name: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integer token IDs.")
    try:
        values = tuple(token_ids)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integer token IDs.") from error

    normalized: list[int] = []
    for position, token_id in enumerate(values):
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError(
                f"{name}[{position}] must be an integer token ID, "
                f"got {type(token_id).__name__}."
            )
        integer = int(token_id)
        if integer < 0:
            raise ValueError(
                f"{name}[{position}] must be non-negative, got {integer}."
            )
        normalized.append(integer)
    return tuple(normalized)


def _validate_weights(weights: Sequence[int]) -> SegmentLengths:
    if isinstance(weights, (str, bytes)):
        raise TypeError("budget_weights must contain exactly three integers.")
    try:
        values = tuple(weights)
    except TypeError as error:
        raise TypeError("budget_weights must contain exactly three integers.") from error
    if len(values) != 3:
        raise ValueError("budget_weights must contain prompt, response_a, response_b.")

    checked = tuple(
        _strict_non_negative_int(value, name=f"budget_weights[{index}]")
        for index, value in enumerate(values)
    )
    if any(value == 0 for value in checked):
        raise ValueError("All budget_weights must be positive.")
    if checked[1] != checked[2]:
        raise ValueError(
            "response_a and response_b budget weights must be equal to avoid "
            "position bias."
        )
    return checked  # type: ignore[return-value]


def _largest_remainder(total: int, weights: Sequence[int]) -> list[int]:
    """Deterministically apportion ``total`` according to integer weights."""
    weight_sum = sum(weights)
    shares = [(total * weight) // weight_sum for weight in weights]
    remainder = total - sum(shares)
    ranking = sorted(
        range(len(weights)),
        key=lambda index: (-(total * weights[index] % weight_sum), index),
    )
    for index in ranking[:remainder]:
        shares[index] += 1
    return shares


def _allocate_content_budget(
    lengths: SegmentLengths,
    content_budget: int,
    weights: SegmentLengths,
) -> SegmentLengths:
    """Allocate all useful budget and redistribute capacity left by short fields."""
    if sum(lengths) <= content_budget:
        return lengths
    if content_budget == 0:
        return (0, 0, 0)

    allocation = [0, 0, 0]
    remaining = content_budget
    active = {index for index, length in enumerate(lengths) if length > 0}

    while remaining > 0 and active:
        ordered = sorted(active)
        shares = _largest_remainder(remaining, [weights[index] for index in ordered])
        distributed = 0
        for index, share in zip(ordered, shares):
            capacity = lengths[index] - allocation[index]
            addition = min(share, capacity)
            allocation[index] += addition
            distributed += addition

        remaining -= distributed
        active = {
            index for index in active if allocation[index] < lengths[index]
        }
        if distributed == 0:
            raise RuntimeError("Budget allocation failed to make progress.")

    result = tuple(allocation)
    if sum(result) > content_budget:
        raise RuntimeError("Allocated content exceeds the available budget.")
    if sum(result) != min(sum(lengths), content_budget):
        raise RuntimeError("Useful content budget was not fully allocated.")
    return result  # type: ignore[return-value]


def _head_tail(token_ids: tuple[int, ...], keep: int) -> list[int]:
    if keep >= len(token_ids):
        return list(token_ids)
    if keep == 0:
        return []
    head = (keep + 1) // 2
    tail = keep - head
    if tail == 0:
        return list(token_ids[:head])
    return [*token_ids[:head], *token_ids[-tail:]]


def balanced_head_tail_truncate(
    prompt_token_ids: Iterable[int],
    response_a_token_ids: Iterable[int],
    response_b_token_ids: Iterable[int],
    *,
    max_length: int,
    special_tokens: int,
    budget_weights: Sequence[int] = (1, 2, 2),
) -> tuple[list[int], list[int], list[int], TruncationStats]:
    """Truncate three token sequences within one strict model-length budget.

    The default weights reserve 20% of content capacity for the prompt and 40%
    for each answer.  Capacity unused by a short segment is redistributed among
    segments that still need tokens.  Every truncated segment keeps both its
    beginning and end; when an odd number is retained, the extra token is kept
    at the beginning.

    ``special_tokens`` is a count, not a token-ID sequence.  It must include all
    tokens the caller will add around/between the returned content sequences.
    """
    checked_max_length = _strict_non_negative_int(max_length, name="max_length")
    if checked_max_length == 0:
        raise ValueError("max_length must be positive.")
    checked_special_tokens = _strict_non_negative_int(
        special_tokens, name="special_tokens"
    )
    if checked_special_tokens > checked_max_length:
        raise ValueError(
            "special_tokens cannot exceed max_length: "
            f"{checked_special_tokens} > {checked_max_length}."
        )
    checked_weights = _validate_weights(budget_weights)

    prompt = _validate_token_ids(prompt_token_ids, name="prompt_token_ids")
    response_a = _validate_token_ids(
        response_a_token_ids, name="response_a_token_ids"
    )
    response_b = _validate_token_ids(
        response_b_token_ids, name="response_b_token_ids"
    )
    sequences = (prompt, response_a, response_b)
    original_lengths: SegmentLengths = tuple(len(item) for item in sequences)  # type: ignore[assignment]
    content_budget = checked_max_length - checked_special_tokens
    allocated_lengths = _allocate_content_budget(
        original_lengths, content_budget, checked_weights
    )
    outputs = tuple(
        _head_tail(token_ids, keep)
        for token_ids, keep in zip(sequences, allocated_lengths)
    )
    output_lengths: SegmentLengths = tuple(len(item) for item in outputs)  # type: ignore[assignment]
    used_content_tokens = sum(output_lengths)

    if output_lengths != allocated_lengths:
        raise RuntimeError("Truncation output does not match the allocation plan.")
    if used_content_tokens + checked_special_tokens > checked_max_length:
        raise RuntimeError("Truncation result exceeds max_length.")

    stats = TruncationStats(
        max_length=checked_max_length,
        special_tokens=checked_special_tokens,
        content_budget=content_budget,
        budget_weights=checked_weights,
        original_lengths=original_lengths,
        allocated_lengths=allocated_lengths,
        output_lengths=output_lengths,
        truncated=tuple(
            output < original
            for output, original in zip(output_lengths, original_lengths)
        ),  # type: ignore[arg-type]
        used_content_tokens=used_content_tokens,
        unused_content_budget=content_budget - used_content_tokens,
    )
    return outputs[0], outputs[1], outputs[2], stats
