#!/usr/bin/env python3
"""Backfill the completed E2 QLoRA HPO into ClearML as historical evidence.

The H100 worker finished the five pre-registered trials before ClearML was
available in that environment.  This script creates a separate, explicitly
historical ClearML task from the versioned summary; it never changes the source
run metadata or represents the import as live H100 tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "docs/evidence/E20260909170000000000_hpo_summary.json"
PROJECT_NAME = "PMLDL LLM Classification Finetuning"
TASK_NAME = "E2 HPO historical import — epoch-aware QLoRA search"


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict) or not isinstance(summary.get("trials"), list):
        raise ValueError(f"Invalid HPO summary: {path}")
    return summary


def get_task(task_id: str | None) -> Any:
    from clearml import Task

    if task_id:
        return Task.get_task(task_id=task_id)
    return Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        tags=["historical-import", "E2", "HPO", "qlora", "H100", "kaggle"],
        reuse_last_task_id=False,
    )


def report_trials(task: Any, summary: dict[str, Any]) -> None:
    logger = task.get_logger()
    for iteration, trial in enumerate(summary["trials"], start=1):
        name = str(trial["id"])
        logger.report_scalar(
            title="selection_fold_7", series="best_log_loss", value=float(trial["best_log_loss"]), iteration=iteration
        )
        logger.report_scalar(
            title="trial_parameters", series=f"rank/{name}", value=float(trial["r"]), iteration=iteration
        )
        logger.report_scalar(
            title="trial_parameters", series=f"learning_rate/{name}", value=float(trial["learning_rate"]), iteration=iteration
        )
        logger.report_scalar(
            title="trial_parameters", series=f"best_epoch/{name}", value=float(trial["best_epoch"]), iteration=iteration
        )

    winner = next(trial for trial in summary["trials"] if trial["id"] == summary["selected_trial"])
    for row in winner.get("epoch_history", []):
        epoch = int(row["epoch"])
        for metric in ("log_loss", "accuracy", "macro_f1"):
            logger.report_scalar(
                title="winner_epoch_history",
                series=metric,
                value=float(row[metric]),
                iteration=epoch,
            )
    for metric in ("best_log_loss", "best_accuracy", "best_macro_f1", "best_ece_15", "best_brier_score", "best_swap_error_l1"):
        logger.report_scalar(
            title="winner_fold_7",
            series=metric.removeprefix("best_"),
            value=float(winner[metric]),
            iteration=int(winner["best_epoch"]),
        )
    kaggle = summary["kaggle"]
    logger.report_scalar(
        title="kaggle_public",
        series="log_loss_fixed_58pct_hpo_qlora_42pct_sparse",
        value=float(kaggle["public_log_loss"]),
        iteration=1,
    )
    logger.report_scalar(
        title="kaggle_public",
        series="log_loss_previous_fixed_blend",
        value=float(kaggle["previous_fixed_blend_public_log_loss"]),
        iteration=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--task-id", help="Populate an existing historical task instead of creating one.")
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    summary = load_summary(summary_path)
    protocol = summary["protocol"]
    kaggle = summary["kaggle"]
    task = get_task(args.task_id)
    task.set_name(TASK_NAME)
    task.set_tags(["historical-import", "E2", "HPO", "qlora", "H100", "kaggle"])
    task.set_parameters(
        {
            "source/experiment_id": str(summary["experiment_id"]),
            "source/parent_experiment_id": str(summary["parent_experiment_id"]),
            "source/import_type": "historical import; not live H100 tracking",
            "source/nirvana_workflow": str(summary["source"]["workflow"]),
            "protocol/seed": int(protocol["seed"]),
            "protocol/train_folds": "0-6",
            "protocol/selection_fold": int(protocol["selection_fold"]),
            "protocol/calibration_fold": "unopened (8)",
            "protocol/final_holdout_fold": "unopened (9)",
            "protocol/max_epochs": int(protocol["max_epochs"]),
            "protocol/model": str(protocol["model"]),
            "protocol/gpu": str(protocol["gpu"]),
            "external/kaggle_notebook": str(kaggle["notebook"]),
            "external/kaggle_submission_ref": str(kaggle["submission_ref"]),
            "external/kaggle_public_log_loss": float(kaggle["public_log_loss"]),
        }
    )
    report_trials(task, summary)
    task.upload_artifact(name="hpo_summary.json", artifact_object=summary_path)
    task.upload_artifact(name="experiment_config.json", artifact_object=ROOT / "configs/experiments/E20260909170000000000.json")
    task.set_comment(
        "Historical import of five completed H100 QLoRA trials. The H100 worker was not "
        "connected to ClearML during training; this task uploads the retained trial metrics "
        "and versioned evidence afterwards. Kaggle external evaluation: " + str(kaggle["notebook"])
    )
    task.mark_completed(force=True, status_message="Historical HPO import completed: source metrics and configuration uploaded.")
    print(f"Imported {summary['experiment_id']} into ClearML task {task.id}")


if __name__ == "__main__":
    main()
