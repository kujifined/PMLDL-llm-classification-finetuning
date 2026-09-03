from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .data import decode_turns


SIDE_STAT_NAMES = (
    "chars",
    "words",
    "lines",
    "turns",
    "null_turns",
    "empty_turns",
    "code_fences",
    "urls",
    "digits",
    "non_ascii_ratio",
)


def _conversation_stats(value: Any) -> dict[str, float]:
    raw_turns = decode_turns(value)
    turns = ["" if item is None else str(item) for item in raw_turns]
    text = "\n".join(turns)
    char_count = len(text)
    return {
        "chars": float(char_count),
        "words": float(len(text.split())),
        "lines": float(text.count("\n") + 1),
        "turns": float(len(turns)),
        "null_turns": float(sum(item is None for item in raw_turns)),
        "empty_turns": float(sum(not turn.strip() for turn in turns)),
        "code_fences": float(text.count(chr(96) * 3)),
        "urls": float(len(re.findall(r"https?://", text, flags=re.IGNORECASE))),
        "digits": float(sum(character.isdigit() for character in text)),
        "non_ascii_ratio": (
            float(sum(ord(character) > 127 for character in text)) / max(char_count, 1)
        ),
    }


def build_bias_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create a small, interpretable feature set without model identities."""
    rows: list[list[float]] = []
    feature_names: list[str] | None = None

    for prompt, response_a, response_b in frame[
        ["prompt", "response_a", "response_b"]
    ].itertuples(index=False, name=None):
        prompt_stats = _conversation_stats(prompt)
        a_stats = _conversation_stats(response_a)
        b_stats = _conversation_stats(response_b)

        features: dict[str, float] = {
            "prompt_chars": prompt_stats["chars"],
            "prompt_words": prompt_stats["words"],
            "prompt_turns": prompt_stats["turns"],
            "prompt_non_ascii_ratio": prompt_stats["non_ascii_ratio"],
        }
        for name in SIDE_STAT_NAMES:
            a_value = a_stats[name]
            b_value = b_stats[name]
            features[f"a_{name}"] = a_value
            features[f"b_{name}"] = b_value
            features[f"diff_{name}"] = a_value - b_value
            features[f"abs_diff_{name}"] = abs(a_value - b_value)
            features[f"log_ratio_{name}"] = math.log1p(max(a_value, 0.0)) - math.log1p(
                max(b_value, 0.0)
            )
        features["combined_response_chars"] = a_stats["chars"] + b_stats["chars"]
        features["max_response_chars"] = max(a_stats["chars"], b_stats["chars"])

        if feature_names is None:
            feature_names = list(features)
        rows.append([features[name] for name in feature_names])

    if feature_names is None:
        feature_names = []
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Bias feature matrix contains non-finite values.")
    return matrix, feature_names

