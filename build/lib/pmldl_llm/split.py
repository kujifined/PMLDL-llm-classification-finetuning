from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import (
    DataContractError,
    file_sha256,
    prompt_group_ids,
    target_indices,
    validate_train_frame,
)


def make_group_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 20260903,
) -> pd.DataFrame:
    """Assign deterministic folds while keeping normalized prompts together."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    validate_train_frame(frame)
    y = target_indices(frame)
    groups = prompt_group_ids(frame["prompt"])
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = np.full(len(frame), -1, dtype=np.int16)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(frame)), y, groups)
    ):
        folds[validation_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("At least one row did not receive a validation fold.")

    assignment = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "prompt_group": groups,
            "fold": folds,
        }
    )
    if assignment.groupby("prompt_group")["fold"].nunique().max() != 1:
        raise RuntimeError("Prompt-group leakage detected across folds.")
    return assignment


def load_frozen_folds(
    frame: pd.DataFrame,
    path: str | Path,
    split_config_path: str | Path,
    n_splits: int | None = None,
    *,
    metadata_path: str | Path | None = None,
    dataset_hashes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load and fully verify an immutable id-to-fold assignment."""
    validate_train_frame(frame)
    fold_path = Path(path)
    if not fold_path.is_file():
        raise FileNotFoundError(
            f"Frozen fold assignment is missing: {fold_path}. Run scripts/freeze_folds.py once."
        )
    split_config_path = Path(split_config_path)
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else fold_path.with_name("metadata.json")
    )
    try:
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataContractError(
            f"Frozen split metadata is missing: {resolved_metadata_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataContractError(
            f"Frozen split metadata is invalid JSON: {resolved_metadata_path}."
        ) from exc
    if not isinstance(metadata, dict):
        raise DataContractError("Frozen split metadata must be a JSON object.")
    if metadata.get("folds_sha256") != file_sha256(fold_path):
        raise DataContractError("Frozen folds checksum does not match metadata.")
    if metadata.get("split_config_sha256") != file_sha256(split_config_path):
        raise DataContractError("Split config checksum does not match frozen metadata.")
    if dataset_hashes is not None and metadata.get("dataset_sha256") != dataset_hashes:
        raise DataContractError("Dataset checksums do not match frozen split metadata.")

    assignment = pd.read_csv(fold_path)
    expected_columns = ["id", "prompt_group", "fold"]
    if assignment.columns.tolist() != expected_columns:
        raise DataContractError(
            f"Frozen folds must have columns {expected_columns}, got {assignment.columns.tolist()}."
        )
    if assignment["id"].duplicated().any():
        raise DataContractError("Frozen fold ids must be unique.")
    if len(assignment) != len(frame):
        raise DataContractError(
            f"Frozen folds contain {len(assignment)} rows; training data contains {len(frame)}."
        )
    if metadata.get("rows") != len(assignment):
        raise DataContractError("Frozen split row count does not match metadata.")

    aligned = frame[["id"]].merge(
        assignment,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[["prompt_group", "fold"]].isna().any().any():
        raise DataContractError("Frozen folds do not contain exactly the current training ids.")

    expected_groups = prompt_group_ids(frame["prompt"])
    if not np.array_equal(aligned["prompt_group"].to_numpy(), expected_groups):
        raise DataContractError(
            "Frozen prompt groups do not match the current training data/normalization."
        )
    raw_fold_values = aligned["fold"].to_numpy()
    fold_values = raw_fold_values.astype(np.int64)
    if not np.array_equal(fold_values, raw_fold_values):
        raise DataContractError("Frozen fold values must be integers.")
    observed = set(fold_values.tolist())
    if n_splits is not None and observed != set(range(n_splits)):
        raise DataContractError(
            f"Frozen folds are {sorted(observed)}, expected every fold in [0, {n_splits - 1}]."
        )
    if aligned.groupby("prompt_group")["fold"].nunique().max() != 1:
        raise DataContractError("Prompt-group leakage detected in frozen folds.")
    aligned["fold"] = fold_values.astype(np.int16)
    return aligned.loc[:, expected_columns]
