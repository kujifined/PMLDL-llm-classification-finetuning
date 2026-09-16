from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMNS, TEXT_COLUMNS


class DataContractError(ValueError):
    """Raised when competition data violates an expected invariant."""


def decode_turns(value: Any) -> list[Any]:
    """Decode a Kaggle conversation cell while preserving JSON null values."""
    if isinstance(value, list):
        turns = value
    elif isinstance(value, str):
        try:
            turns = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DataContractError("Conversation cell is not valid JSON.") from exc
    else:
        raise DataContractError(
            f"Conversation cell must be a JSON string or list, got {type(value).__name__}."
        )
    if not isinstance(turns, list):
        raise DataContractError("Conversation cell must decode to a JSON list.")
    return turns


def clean_turns(value: Any) -> list[str]:
    """Decode turns and replace JSON null with an explicit empty string."""
    cleaned: list[str] = []
    for turn in decode_turns(value):
        if turn is None:
            cleaned.append("")
        elif isinstance(turn, str):
            cleaned.append(turn)
        else:
            cleaned.append(json.dumps(turn, ensure_ascii=False, sort_keys=True))
    return cleaned


def canonical_prompt(value: Any) -> str:
    """Normalize a prompt for leakage-safe grouping of repeated prompts."""
    normalized = []
    for turn in clean_turns(value):
        turn = unicodedata.normalize("NFKC", turn)
        turn = re.sub(r"\s+", " ", turn).strip().casefold()
        normalized.append(turn)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def prompt_group_ids(values: Iterable[Any]) -> np.ndarray:
    groups = []
    for value in values:
        canonical = canonical_prompt(value)
        groups.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20])
    return np.asarray(groups)


def target_indices(frame: pd.DataFrame) -> np.ndarray:
    targets = frame.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.int64)
    return targets.argmax(axis=1)


def validate_train_frame(frame: pd.DataFrame) -> None:
    required = {"id", "model_a", "model_b", *TEXT_COLUMNS, *TARGET_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"Training data is missing columns: {sorted(missing)}")
    if frame.loc[:, list(required)].isna().any().any():
        raise DataContractError("Training data contains CSV-level missing values.")
    if not frame["id"].is_unique:
        raise DataContractError("Training ids must be unique.")

    targets = frame.loc[:, list(TARGET_COLUMNS)]
    if not targets.isin([0, 1]).all().all():
        raise DataContractError("Targets must be binary one-hot columns.")
    if not targets.sum(axis=1).eq(1).all():
        raise DataContractError("Every training row must have exactly one winning class.")

    for values in frame.loc[:, TEXT_COLUMNS].itertuples(index=False, name=None):
        turn_counts = [len(decode_turns(value)) for value in values]
        if len(set(turn_counts)) != 1:
            raise DataContractError(
                "Prompt and both responses must have the same number of turns."
            )


def validate_test_frame(frame: pd.DataFrame) -> None:
    required = {"id", *TEXT_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"Test data is missing columns: {sorted(missing)}")
    if frame.loc[:, list(required)].isna().any().any():
        raise DataContractError("Test data contains CSV-level missing values.")
    if not frame["id"].is_unique:
        raise DataContractError("Test ids must be unique.")
    for values in frame.loc[:, TEXT_COLUMNS].itertuples(index=False, name=None):
        turn_counts = [len(decode_turns(value)) for value in values]
        if len(set(turn_counts)) != 1:
            raise DataContractError(
                "Prompt and both responses must have the same number of turns."
            )


def load_competition_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv under {data_dir.resolve()}."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_train_frame(train)
    validate_test_frame(test)
    return train, test


def swapped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Swap response A/B and, when present, the corresponding model/target columns."""
    swapped = frame.copy()
    swapped[["response_a", "response_b"]] = frame[
        ["response_b", "response_a"]
    ].to_numpy()
    if {"model_a", "model_b"}.issubset(frame.columns):
        swapped[["model_a", "model_b"]] = frame[["model_b", "model_a"]].to_numpy()
    if set(TARGET_COLUMNS).issubset(frame.columns):
        swapped[["winner_model_a", "winner_model_b"]] = frame[
            ["winner_model_b", "winner_model_a"]
        ].to_numpy()
    return swapped


def swap_target_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Target indices must be 0 (A), 1 (B), or 2 (tie).")
    return np.choose(values, [1, 0, 2])


def swap_probability_columns(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Expected a probability matrix with shape (n_rows, 3).")
    return probabilities[:, [1, 0, 2]]


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksum_manifest(path: str | Path) -> dict[str, str]:
    """Parse a sha256sum-compatible manifest with safe relative paths."""
    manifest_path = Path(path)
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DataContractError(
                f"Invalid checksum manifest line {line_number}: {raw_line!r}."
            )
        digest, relative_name = parts
        relative_name = relative_name.lstrip("*")
        relative_path = Path(relative_name)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise DataContractError(
                f"Unsafe or invalid checksum entry on line {line_number}."
            )
        normalized_name = relative_path.as_posix()
        if normalized_name in entries:
            raise DataContractError(
                f"Duplicate checksum entry for {normalized_name!r}."
            )
        entries[normalized_name] = digest.lower()
    if not entries:
        raise DataContractError("Checksum manifest contains no file entries.")
    return entries


def verify_checksum_manifest(path: str | Path) -> dict[str, str]:
    """Verify every file in a checksum manifest and return its expected hashes."""
    manifest_path = Path(path)
    entries = load_checksum_manifest(manifest_path)
    mismatches: list[str] = []
    for relative_name, expected in entries.items():
        file_path = manifest_path.parent / relative_name
        if not file_path.is_file():
            mismatches.append(f"missing: {relative_name}")
            continue
        actual = file_sha256(file_path)
        if actual != expected:
            mismatches.append(
                f"sha256 mismatch: {relative_name} (expected {expected}, got {actual})"
            )
    if mismatches:
        raise DataContractError("Dataset checksum verification failed: " + "; ".join(mismatches))
    return entries


def verify_competition_data_dir(
    data_dir: str | Path,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    """Bind train/test/sample hashes to the exact directory a command will read."""
    directory = Path(data_dir)
    verified: dict[str, str] = {}
    for filename in ("train.csv", "test.csv", "sample_submission.csv"):
        matches = [
            (relative_name, digest)
            for relative_name, digest in expected_hashes.items()
            if Path(relative_name).name == filename
        ]
        if len(matches) != 1:
            raise DataContractError(
                f"Checksum manifest must contain exactly one entry named {filename!r}."
            )
        relative_name, expected = matches[0]
        actual_path = directory / filename
        if not actual_path.is_file():
            raise DataContractError(f"Competition data file is missing: {actual_path}.")
        actual = file_sha256(actual_path)
        if actual != expected:
            raise DataContractError(
                f"Loaded data checksum mismatch for {actual_path}: "
                f"expected {expected}, got {actual}."
            )
        verified[relative_name] = actual
    return verified
