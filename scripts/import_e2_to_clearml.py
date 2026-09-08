#!/usr/bin/env python3
"""Import the completed E2 run into ClearML as an explicitly historical task.

The H100 worker that produced E2 could not initialise the ClearML client.  This
script uploads the immutable local record afterwards; it never changes the
original ``run.json`` tracking state.  Pass an existing task id to populate the
task created in the shared project, or omit it to create a new task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / (
    "results/runs/E20260905212645934620__s42__38aea1ac__20260907T052625009035Z"
)
PROJECT_NAME = "PMLDL LLM Classification Finetuning"
TASK_NAME = "E2 historical import — DeBERTa Full FT vs LoRA vs QLoRA"
KAGGLE_NOTEBOOK = (
    "https://www.kaggle.com/code/karimkhabibrakhmanov/"
    "pmldl-e2-fixed-qlora-sparse-blend?scriptVersionId=348053027"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def get_task(task_id: str | None):
    from clearml import Task

    if task_id:
        return Task.get_task(task_id=task_id)
    return Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        tags=["historical-import", "E2", "qlora", "lora", "full-finetuning"],
        reuse_last_task_id=False,
    )


def upload_history(task: Any, history: list[dict[str, Any]]) -> None:
    logger = task.get_logger()
    for row in history:
        namespace = str(row["namespace"])
        step = int(row["step"])
        for name, value in row["metrics"].items():
            logger.report_scalar(
                title=namespace,
                series=str(name),
                value=float(value),
                iteration=step,
            )


def upload_summary(task: Any, summary: dict[str, Any]) -> None:
    logger = task.get_logger()
    for namespace, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                logger.report_scalar(
                    title=namespace,
                    series=str(name),
                    value=float(value),
                    iteration=0,
                )
    logger.report_scalar(
        title="kaggle_public",
        series="log_loss_fixed_58pct_qlora_42pct_sparse",
        value=1.02067,
        iteration=0,
    )
    logger.report_scalar(
        title="kaggle_public",
        series="log_loss_qlora_only",
        value=1.02832,
        iteration=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Saved E2 run directory."
    )
    parser.add_argument(
        "--task-id",
        help="Existing ClearML task to populate; omit to create a new historical task.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    run_path = run_dir / "run.json"
    config_path = run_dir / "config.json"
    for path in (metrics_path, run_path, config_path):
        if not path.is_file():
            raise SystemExit(f"Missing E2 source artifact: {path}")

    metrics = read_json(metrics_path)
    run = read_json(run_path)
    task = get_task(args.task_id)
    task.set_name(TASK_NAME)
    task.set_tags(["historical-import", "E2", "qlora", "lora", "full-finetuning", "H100"])
    task.set_parameters(
        {
            "source/run_id": str(metrics["run_id"]),
            "source/experiment_id": str(metrics["experiment_id"]),
            "source/tracking_status": str(run["tracking"]["status"]),
            "source/import_type": "historical import; not live H100 tracking",
            "protocol/seed": int(metrics["seed"]),
            "protocol/train_folds": "0-6",
            "protocol/validation_fold": 7,
            "protocol/epochs": 3,
            "protocol/model": "microsoft/deberta-v3-base",
            "external/kaggle_notebook": KAGGLE_NOTEBOOK,
            "external/kaggle_public_log_loss_fixed_blend": 1.02067,
        }
    )
    upload_history(task, metrics["history"])
    upload_summary(task, metrics["summary"])
    for path in (config_path, metrics_path, run_path):
        task.upload_artifact(name=path.name, artifact_object=path)
    task.set_comment(
        "Historical import of completed E2. The original H100 run has "
        "tracking_status=unavailable; this task reconstructs its retained local "
        "history and artifacts. Kaggle notebook: " + KAGGLE_NOTEBOOK
    )
    task.close()
    print(f"Imported {metrics['run_id']} into ClearML task {task.id}")


if __name__ == "__main__":
    main()
