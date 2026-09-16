#!/usr/bin/env python3
"""Publish the frozen seed-42 fold-8 calibration handoff to ClearML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "artifacts/final_inputs/deberta_qlora_seed42"
PROJECT_NAME = "PMLDL LLM Classification Finetuning"
TASK_NAME = "Sprint 2 E2 fold-8 handoff — seed-42 QLoRA"
EXPECTED_COLUMNS = [
    "id",
    "target",
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_handoff(prediction_path: Path, manifest: dict[str, Any]) -> int:
    import numpy as np
    import pandas as pd

    predictions = pd.read_csv(prediction_path)
    if predictions.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns: {predictions.columns.tolist()} != {EXPECTED_COLUMNS}"
        )
    if predictions.empty or predictions["id"].duplicated().any():
        raise ValueError("Fold-8 predictions must have non-empty, unique IDs.")
    probabilities = predictions[EXPECTED_COLUMNS[2:]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Fold-8 probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("Fold-8 probabilities must sum to one.")

    artifact = manifest["artifacts"]["fold8_predictions"]
    if artifact["sha256"] != sha256(prediction_path):
        raise ValueError("Prediction SHA256 differs from the manifest.")
    if int(artifact["rows"]) != len(predictions):
        raise ValueError("Prediction row count differs from the manifest.")
    if artifact["columns"] != EXPECTED_COLUMNS:
        raise ValueError("Prediction columns differ from the manifest.")
    if manifest["protocol"]["calibration_fold"] != 8:
        raise ValueError("Expected calibration fold 8.")
    if manifest["protocol"]["final_holdout_fold_unopened"] != 9:
        raise ValueError("Expected unopened final holdout fold 9.")
    if manifest["fold8_inference"]["fold9_accessed"] is not False:
        raise ValueError("The final holdout must remain unopened.")
    return len(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--task-id", help="Update an existing task instead of creating one.")
    parser.add_argument("--nirvana-process", required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    prediction_path = input_dir / "fold8_predictions.csv"
    manifest_path = input_dir / "model_manifest.json"
    manifest = read_json(manifest_path)
    row_count = validate_handoff(prediction_path, manifest)

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
                "fold-8",
                "calibration-handoff",
                "H100",
            ],
            reuse_last_task_id=False,
        )
    )
    task.set_name(TASK_NAME)
    task.set_parameters(
        {
            "source/run_id": manifest["run_id"],
            "source/training_git_commit": manifest["git_commit"],
            "source/inference_git_commit": manifest["fold8_inference"]["code_commit"],
            "source/nirvana_process": args.nirvana_process,
            "source/adapter_sha256": manifest["fold8_inference"]["source_adapter_sha256"],
            "handoff/fold": 8,
            "handoff/rows": row_count,
            "handoff/prediction_sha256": sha256(prediction_path),
            "handoff/manifest_sha256": sha256(manifest_path),
            "handoff/schema": ",".join(EXPECTED_COLUMNS),
            "protocol/final_holdout_fold": "unopened (9)",
            "runtime/gpu": manifest["fold8_inference"]["gpu"],
            "runtime/compute_dtype": manifest["fold8_inference"]["compute_dtype"],
        }
    )
    task.upload_artifact(
        name="fold8_predictions.csv",
        artifact_object=prediction_path,
        metadata={
            "sha256": sha256(prediction_path),
            "rows": row_count,
            "schema": EXPECTED_COLUMNS,
        },
        wait_on_upload=True,
    )
    task.upload_artifact(
        name="model_manifest.json",
        artifact_object=manifest_path,
        metadata={"sha256": sha256(manifest_path)},
        wait_on_upload=True,
    )
    task.set_comment(
        "Inference-only calibration handoff from the fold-7-selected seed-42 "
        "DeBERTa QLoRA checkpoint. The frozen adapter SHA256 was verified before "
        "inference. Fold 9 remained unopened."
    )
    task.flush(wait_for_uploads=True)
    task.mark_completed(
        force=True,
        status_message="Fold-8 predictions and manifest validated and uploaded.",
    )
    task.reload()
    artifacts = {
        name: artifact.url
        for name, artifact in task.artifacts.items()
        if name in {"fold8_predictions.csv", "model_manifest.json"}
    }
    print(
        json.dumps(
            {
                "task_id": task.id,
                "task_url": task.get_output_log_web_page(),
                "artifacts": artifacts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
