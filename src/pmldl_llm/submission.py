from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMNS
from .evaluation import normalize_probabilities


def build_submission(ids: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    probabilities = normalize_probabilities(probabilities)
    ids = np.asarray(ids)
    if len(ids) != len(probabilities):
        raise ValueError("Number of ids and prediction rows must match.")
    if pd.Series(ids).duplicated().any():
        raise ValueError("Submission ids must be unique.")
    submission = pd.DataFrame(probabilities, columns=TARGET_COLUMNS)
    submission.insert(0, "id", ids)
    return submission


def write_submission(
    ids: np.ndarray,
    probabilities: np.ndarray,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = build_submission(ids, probabilities)
    submission.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    if list(reloaded.columns) != ["id", *TARGET_COLUMNS]:
        raise RuntimeError("Written submission has an invalid column order.")
    if not np.allclose(reloaded.loc[:, list(TARGET_COLUMNS)].sum(axis=1), 1.0):
        raise RuntimeError("Written submission probabilities do not sum to one.")
    return path
