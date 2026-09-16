#!/usr/bin/env python3
"""Check that the tracked final-release metadata is internally consistent."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required release file is missing: {path}")


def main() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "configs" / "final_model.json",
        ROOT / "configs" / "final_model_search.json",
        ROOT / "results" / "model_comparison.csv",
        ROOT / "results" / "final" / "ensemble_grid.csv",
        ROOT / "results" / "final" / "final_comparison.csv",
        ROOT / "docs" / "REPRODUCE.md",
        ROOT / "artifacts" / "FINAL_ARTIFACTS.md",
    )
    for path in required:
        require(path)

    final_model = json.loads(
        (ROOT / "configs" / "final_model.json").read_text(encoding="utf-8")
    )
    if final_model.get("status") != "frozen":
        raise ValueError("configs/final_model.json must have status 'frozen'.")
    names = [model.get("name") for model in final_model.get("models", [])]
    if names != ["deberta_qlora_seed42", "sparse_tfidf_sprint2"]:
        raise ValueError("Final model must contain the two declared frozen models.")
    weights = [float(model.get("weight", 0.0)) for model in final_model["models"]]
    if len(weights) != 2 or any(weight <= 0 for weight in weights):
        raise ValueError("Final model weights must be positive.")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("Final model weights must sum to one.")

    with (ROOT / "results" / "model_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not any(
        row.get("model") == "Frozen DeBERTa plus sparse ensemble" for row in rows
    ):
        raise ValueError("results/model_comparison.csv lacks the final ensemble row.")
    print("Release metadata is internally consistent.")


if __name__ == "__main__":
    main()
