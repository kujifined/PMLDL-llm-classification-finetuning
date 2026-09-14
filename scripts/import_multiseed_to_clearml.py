#!/usr/bin/env python3
"""Backfill the Sprint 2 E2 multi-seed study into ClearML.

The three H100 jobs completed without the ClearML package in their runtime.
This importer preserves that fact and uploads the retained metrics, prediction
manifests, adapters and fold-7 probabilities as historical evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/e2_multiseed/summary.json"
PROJECT_NAME = "PMLDL LLM Classification Finetuning"
TASK_NAME = "Sprint 2 E2 historical import — fixed three-seed QLoRA"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def get_task(task_id: str | None) -> Any:
    from clearml import Task

    if task_id:
        return Task.get_task(task_id=task_id)
    return Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        tags=[
            "historical-import",
            "sprint-2",
            "E2",
            "qlora",
            "multiseed",
            "H100",
            "selection-only",
        ],
        reuse_last_task_id=False,
    )


def report_member(task: Any, metrics: dict[str, Any]) -> None:
    logger = task.get_logger()
    seed = int(metrics["seed"])
    for row in metrics["history"]:
        epoch = int(row["step"])
        for metric, value in row["metrics"].items():
            logger.report_scalar(
                title=f"seed_{seed}_epoch_history",
                series=str(metric),
                value=float(value),
                iteration=epoch,
            )
    for metric, value in metrics["summary"]["validation"].items():
        logger.report_scalar(
            title="fold_7_single_seed",
            series=f"{metric}/seed_{seed}",
            value=float(value),
            iteration=seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--task-id",
        help="Populate an existing historical task instead of creating one.",
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    summary = read_json(summary_path)
    task = get_task(args.task_id)
    task.set_name(TASK_NAME)
    task.set_tags(
        [
            "historical-import",
            "sprint-2",
            "E2",
            "qlora",
            "multiseed",
            "H100",
            "selection-only",
        ]
    )
    source = summary["source"]
    protocol = summary["protocol"]
    task.set_parameters(
        {
            "source/import_type": "historical import; not live H100 tracking",
            "source/nirvana_process": str(source["nirvana_process"]),
            "source/git_commit": str(source["git_commit"]),
            "protocol/model": str(protocol["model"]),
            "protocol/model_revision": str(protocol["model_revision"]),
            "protocol/train_folds": "0-6",
            "protocol/selection_fold": int(summary["selection_fold"]),
            "protocol/epochs": int(protocol["epochs"]),
            "protocol/seeds": ",".join(map(str, protocol["seeds"])),
            "protocol/calibration_fold": str(summary["fold8_status"]) + " (8)",
            "protocol/final_holdout_fold": "unopened (9)",
            "selection/recommended_candidate": str(summary["recommended_candidate"]),
            "selection/recommendation_log_loss": float(summary["recommendation_log_loss"]),
            "selection/kaggle_used": bool(summary["kaggle_used_for_selection"]),
        }
    )

    run_sources: list[tuple[Path, Path]] = []
    for member in summary["members"]:
        run_id = str(member["run_id"])
        run_dir = ROOT / "results/runs" / run_id
        artifact_dir = ROOT / "artifacts" / run_id
        metrics = read_json(run_dir / "metrics.json")
        run = read_json(run_dir / "run.json")
        if run["tracking"]["status"] != "unavailable":
            raise ValueError(f"Unexpected original tracking status for {run_id}")
        report_member(task, metrics)
        run_sources.append((run_dir, artifact_dir))

    logger = task.get_logger()
    for metric, value in summary["three_seed_mean"].items():
        logger.report_scalar(
            title="fold_7_three_seed_mean",
            series=str(metric),
            value=float(value),
            iteration=0,
        )
    logger.report_scalar(
        title="fold_7_selection",
        series="ensemble_delta_vs_best_single",
        value=float(summary["ensemble_delta_vs_best_single"]),
        iteration=0,
    )

    task.upload_artifact(name="multiseed_summary.json", artifact_object=summary_path)
    task.upload_artifact(
        name="fold7_comparison.csv",
        artifact_object=summary_path.with_name("fold7_comparison.csv"),
    )
    task.upload_artifact(
        name="three_seed_model_manifest.json",
        artifact_object=ROOT / "artifacts/e2_multiseed/model_manifest.json",
    )
    task.upload_artifact(
        name="three_seed_fold7_predictions.csv",
        artifact_object=ROOT / "artifacts/e2_multiseed/fold7_predictions.csv",
    )
    for run_dir, artifact_dir in run_sources:
        run_id = run_dir.name
        for filename in ("config.json", "metrics.json", "run.json"):
            task.upload_artifact(
                name=f"{run_id}_{filename}", artifact_object=run_dir / filename
            )
        task.upload_artifact(
            name=f"{run_id}_model_manifest.json",
            artifact_object=artifact_dir / "model_manifest.json",
        )
        task.upload_artifact(
            name=f"{run_id}_fold7_predictions.csv",
            artifact_object=artifact_dir / "predictions/fold7_predictions.csv",
        )
        task.upload_artifact(
            name=f"{run_id}_qlora_adapter.zip",
            artifact_object=artifact_dir / "models/qlora_adapter.zip",
        )

    task.set_comment(
        "Historical import of three clean H100 QLoRA jobs. ClearML was absent "
        "from the execution image, so this task reconstructs the retained epoch "
        "curves and uploads all handoff artifacts after completion. Fold 8, fold "
        "9 and Kaggle were not used for this selection."
    )
    task.mark_completed(
        force=True,
        status_message=(
            "Historical Sprint 2 E2 import completed: three seeds and handoff "
            "artifacts uploaded."
        ),
    )
    print(f"Imported Sprint 2 E2 multi-seed evidence into ClearML task {task.id}")


if __name__ == "__main__":
    main()
