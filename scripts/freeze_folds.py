#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.config import validate_fold_roles
from pmldl_llm.constants import (
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_METADATA_PATH,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
)
from pmldl_llm.data import (
    file_sha256,
    load_competition_data,
    verify_checksum_manifest,
    verify_competition_data_dir,
)
from pmldl_llm.split import make_group_folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the immutable id-to-fold assignment used by all experiments."
    )
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_FOLD_METADATA_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing assignment only after it has been regenerated explicitly.",
    )
    args = parser.parse_args()

    if (args.output.exists() or args.metadata.exists()) and not args.force:
        raise FileExistsError(
            "Frozen split already exists. Refusing to overwrite it without --force."
        )
    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    train, _ = load_competition_data(args.data_dir)
    assignment = make_group_folds(
        train,
        n_splits=roles.n_splits,
        seed=roles.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.output, index=False)
    metadata = {
        "schema_version": 1,
        "generator": "sklearn.model_selection.StratifiedGroupKFold",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split_config": split_config,
        "split_config_sha256": file_sha256(args.split_config),
        "dataset_sha256": dataset_hashes,
        "rows": int(len(assignment)),
        "fold_counts": {
            str(key): int(value)
            for key, value in assignment["fold"].value_counts().sort_index().items()
        },
        "folds_sha256": file_sha256(args.output),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"\nFrozen folds saved to {args.output}")


if __name__ == "__main__":
    main()
