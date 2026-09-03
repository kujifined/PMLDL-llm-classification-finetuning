#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.constants import (
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    TARGET_COLUMNS,
    TEXT_COLUMNS,
)
from pmldl_llm.data import (
    canonical_prompt,
    decode_turns,
    file_sha256,
    load_competition_data,
    target_indices,
    verify_competition_data_dir,
    verify_checksum_manifest,
)


def quantiles(values: pd.Series) -> dict[str, float]:
    result = values.quantile([0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {str(index): float(value) for index, value in result.items()}


def conversation_summary(values: pd.Series) -> dict[str, object]:
    turn_counts = []
    char_counts = []
    rows_with_null = 0
    null_items = 0
    for value in values:
        turns = decode_turns(value)
        turn_counts.append(len(turns))
        char_counts.append(sum(len(str(turn)) for turn in turns if turn is not None))
        row_nulls = sum(turn is None for turn in turns)
        rows_with_null += int(row_nulls > 0)
        null_items += row_nulls
    return {
        "turn_count_quantiles": quantiles(pd.Series(turn_counts)),
        "character_count_quantiles": quantiles(pd.Series(char_counts)),
        "rows_with_json_null": rows_with_null,
        "json_null_items": null_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit.json",
    )
    args = parser.parse_args()

    verified_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, verified_hashes)
    train, test = load_competition_data(args.data_dir)
    labels = target_indices(train)
    canonical_prompts = train["prompt"].map(canonical_prompt)
    group_sizes = canonical_prompts.value_counts()
    full_input = train[list(TEXT_COLUMNS)].astype(str).agg("\x1f".join, axis=1)
    label_names = pd.Series(labels).map(dict(enumerate(TARGET_COLUMNS)))
    duplicate_labels = pd.DataFrame({"key": full_input, "label": label_names})

    summary = {
        "data_dir": args.data_dir.name,
        "checksum_manifest_verified": True,
        "checksum_manifest_entries": verified_hashes,
        "files": {
            filename: {
                "bytes": (args.data_dir / filename).stat().st_size,
                "sha256": file_sha256(args.data_dir / filename),
            }
            for filename in ("train.csv", "test.csv", "sample_submission.csv")
        },
        "train": {
            "shape": list(train.shape),
            "columns": train.columns.tolist(),
            "label_counts": {
                TARGET_COLUMNS[index]: int((labels == index).sum())
                for index in range(3)
            },
            "label_rates": {
                TARGET_COLUMNS[index]: float((labels == index).mean())
                for index in range(3)
            },
            "normalized_prompt_groups": int(canonical_prompts.nunique()),
            "rows_in_repeated_prompt_groups": int(
                canonical_prompts.map(group_sizes).gt(1).sum()
            ),
            "largest_prompt_group": int(group_sizes.max()),
            "exact_full_input_duplicate_rows": int(
                full_input.duplicated(keep=False).sum()
            ),
            "conflicting_full_input_groups": int(
                (duplicate_labels.groupby("key")["label"].nunique() > 1).sum()
            ),
            "conversation_columns": {
                column: conversation_summary(train[column]) for column in TEXT_COLUMNS
            },
        },
        "test": {
            "shape": list(test.shape),
            "columns": test.columns.tolist(),
            "note": (
                "The local three-row test is a schema example. Kaggle replaces it "
                "with the hidden scoring set."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved audit to {args.output}")


if __name__ == "__main__":
    main()
