from __future__ import annotations

from typing import Any

import pandas as pd

from .data import decode_turns


def flatten_conversation(
    value: Any,
    null_token: str = " null_response ",
) -> str:
    """Flatten JSON turns while preserving boundaries and missing responses."""
    parts = []
    for index, turn in enumerate(decode_turns(value), start=1):
        content = null_token if turn is None else str(turn)
        parts.append(f" turn_{index}_start {content} turn_{index}_end ")
    return " ".join(parts)


def flattened_text_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    prompts = [flatten_conversation(value) for value in frame["prompt"]]
    responses_a = [flatten_conversation(value) for value in frame["response_a"]]
    responses_b = [flatten_conversation(value) for value in frame["response_b"]]
    return prompts, responses_a, responses_b

