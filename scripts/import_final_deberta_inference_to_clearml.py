#!/usr/bin/env python3
"""Publish the final seed-42 DeBERTa holdout artifacts to ClearML.

The source H100 job has no ClearML package.  This task is deliberately a
post-run artifact import: it keeps the H100 process, frozen source adapter and
CSV checksums attached to the result without claiming that ClearML tracked the
original execution live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "artifacts/final_inputs/deberta_qlora_seed42"
DEFAULT_ADAPTER = (
    ROOT
    / "artifacts/E20260914010000000000__s42__e76c1fcb__20260913T215932089199Z"
    / "models/qlora_adapter.zip"
)
PROJECT_NAME = "PMLDL LLM Classification Finetuning"
TASK_NAME = "Sprint 2 E2 final holdout — seed-42 DeBERTa QLoRA"
PREDICTION_COLUMNS = [
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
]
FOLD_COLUMNS = ["id", "target", *PREDICTION_COLUMNS]
SUBMISSION_COLUMNS = ["id", *PREDICTION_COLUMNS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_probability_frame(path: Path, expected_columns: list[str]) -> int:
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected columns in {path.name}: {frame.columns.tolist()}")
    if frame.empty or frame["id"].duplicated().any():
        raise ValueError(f"{path.name} needs non-empty, unique IDs.")
    probabilities = frame[PREDICTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{path.name} has invalid probability values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError(f"{path.name} probabilities must sum to one.")
    return len(frame)


def validate_inputs(
    input_dir: Path, adapter_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    manifest_path = input_dir / "final_model_manifest.json"
    fold_path = input_dir / "fold9_predictions.csv"
    schema_path = input_dir / "inference_predictions.csv"
    manifest = read_json(manifest_path)

    fold_rows = validate_probability_frame(fold_path, FOLD_COLUMNS)
    schema_rows = validate_probability_frame(schema_path, SUBMISSION_COLUMNS)
    if schema_rows != 3:
        raise ValueError("Local Kaggle schema check must contain exactly three rows.")
    if manifest["protocol"]["target_fold"] != 9:
        raise ValueError("The manifest must declare target fold 9.")
    if manifest["final_inference"]["fold9_accessed"] is not True:
        raise ValueError("The manifest must record the one-time fold-9 access.")
    if manifest["artifacts"]["adapter"]["sha256"] != sha256(adapter_path):
        raise ValueError("Adapter SHA256 differs from the manifest.")
    for key, path, rows, columns in (
        ("fold9_predictions", fold_path, fold_rows, FOLD_COLUMNS),
        ("kaggle_test_predictions", schema_path, schema_rows, SUBMISSION_COLUMNS),
    ):
        artifact = manifest["artifacts"][key]
        if artifact["sha256"] != sha256(path):
            raise ValueError(f"{path.name} SHA256 differs from the manifest.")
        if int(artifact["rows"]) != rows or artifact["columns"] != columns:
            raise ValueError(f"{path.name} metadata differs from the manifest.")

    fold = pd.read_csv(fold_path)
    probabilities = fold[PREDICTION_COLUMNS].to_numpy(dtype=float)
    targets = fold["target"].to_numpy(dtype=int)
    observed = {
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(targets, np.argmax(probabilities, axis=1))),
        "macro_f1": float(
            f1_score(targets, np.argmax(probabilities, axis=1), average="macro")
        ),
    }
    recorded = manifest["final_inference"]["fold9_metrics"]
    for key, value in observed.items():
        if not np.isclose(value, float(recorded[key]), atol=1e-12, rtol=0.0):
            raise ValueError(f"Manifest metric mismatch for {key}.")
    return manifest, {"fold_rows": fold_rows, "schema_rows": schema_rows, **observed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--nirvana-process", required=True)
    parser.add_argument("--task-id", help="Update an existing task instead of creating one.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    adapter_path = args.adapter.resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    manifest, observed = validate_inputs(input_dir, adapter_path)
    manifest_path = input_dir / "final_model_manifest.json"
    fold_path = input_dir / "fold9_predictions.csv"
    schema_path = input_dir / "inference_predictions.csv"

    from clearml import Task

    task = (
        Task.get_task(task_id=args.task_id)
        if args.task_id
        else Task.init(
            project_name=PROJECT_NAME,
            task_name=TASK_NAME,
            task_type=Task.TaskTypes.inference,
            tags=[
                "sprint-2",
                "E2",
                "qlora",
                "seed-42",
                "fold-9",
                "final-holdout",
                "H100",
                "post-run-artifact-import",
            ],
            reuse_last_task_id=False,
        )
    )
    task.set_name(TASK_NAME)
    task.set_parameters(
        {
            "source/tracking": "post-run artifact import; source H100 job was not live-tracked",
            "source/run_id": manifest["run_id"],
            "source/training_git_commit": manifest["git_commit"],
            "source/inference_git_commit": manifest["final_inference"]["code_commit"],
            "source/nirvana_process": args.nirvana_process,
            "source/adapter_sha256": sha256(adapter_path),
            "protocol/selection_fold": manifest["protocol"]["selection_fold"],
            "protocol/calibration_fold": manifest["protocol"]["calibration_fold"],
            "protocol/final_holdout_fold": 9,
            "protocol/swap_averaged_inference": True,
            "fold9/rows": observed["fold_rows"],
            "fold9/log_loss": observed["log_loss"],
            "fold9/accuracy": observed["accuracy"],
            "fold9/macro_f1": observed["macro_f1"],
            "fold9/predictions_sha256": sha256(fold_path),
            "kaggle/local_schema_rows": observed["schema_rows"],
            "kaggle/local_schema_predictions_sha256": sha256(schema_path),
            "runtime/gpu": manifest["final_inference"]["gpu"],
            "runtime/compute_dtype": manifest["final_inference"]["compute_dtype"],
        }
    )
    logger = task.get_logger()
    for key in ("log_loss", "accuracy", "macro_f1"):
        logger.report_scalar(
            title="fold_9_raw_deberta", series=key, value=observed[key], iteration=0
        )
    for name, path, metadata in (
        (
            "fold9_predictions.csv",
            fold_path,
            {"sha256": sha256(fold_path), "rows": observed["fold_rows"], "schema": FOLD_COLUMNS},
        ),
        (
            "kaggle_local_schema_submission.csv",
            schema_path,
            {
                "sha256": sha256(schema_path),
                "rows": observed["schema_rows"],
                "schema": SUBMISSION_COLUMNS,
                "note": "Three-row local schema check; not Kaggle hidden test output.",
            },
        ),
        (
            "final_model_manifest.json",
            manifest_path,
            {"sha256": sha256(manifest_path)},
        ),
        (
            "qlora_adapter.zip",
            adapter_path,
            {"sha256": sha256(adapter_path), "bytes": adapter_path.stat().st_size},
        ),
    ):
        task.upload_artifact(
            name=name, artifact_object=path, metadata=metadata, wait_on_upload=True
        )
    task.set_comment(
        "Inference-only post-run artifact import for the frozen seed-42 DeBERTa "
        "QLoRA checkpoint. Candidate selection used fold 7 and calibration used "
        "fold 8 before this one-time raw-component fold-9 evaluation. The "
        "Kaggle CSV attached here is only the three-row local schema check; it "
        "is not a hidden-test prediction or a competition submission."
    )
    task.flush(wait_for_uploads=True)
    task.mark_completed(
        force=True,
        status_message="Fold-9 CSV, schema check, manifest and adapter validated and uploaded.",
    )
    task.reload()
    artifacts = {
        name: artifact.url
        for name, artifact in task.artifacts.items()
        if name
        in {
            "fold9_predictions.csv",
            "kaggle_local_schema_submission.csv",
            "final_model_manifest.json",
            "qlora_adapter.zip",
        }
    }
    print(
        json.dumps(
            {
                "task_id": task.id,
                "task_url": task.get_output_log_web_page(),
                "artifacts": artifacts,
                "fold9_metrics": {
                    key: observed[key] for key in ("log_loss", "accuracy", "macro_f1")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
