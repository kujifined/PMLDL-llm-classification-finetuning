from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

from .experiment import (
    ALLOWED_TRACKS,
    EXPERIMENT_ID_PATTERN,
    ExperimentRun,
    validate_experiment_config,
)
from .result_collection import collect_result_rows, write_leaderboard
from .run_validation import validate_run_directory


@dataclass
class NotebookExperimentSetup:
    experiment_id: str
    owner: str
    title: str
    hypothesis: str
    changed_factor: str
    model_name: str | None
    track: str = "neural"
    model_revision: str | None = None
    parent_experiment_id: str | None = None
    seed: int = 42
    evaluation_role: str | None = None
    smoke_test: bool = True
    training: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    tracking_mode: str | None = None


@dataclass
class ExperimentOutput:
    metrics: Mapping[str, Real]
    artifacts: Mapping[str, str | Path] = field(default_factory=dict)

    @classmethod
    def from_predictions(
        cls,
        *,
        y_true: Any,
        original_probabilities: Any,
        swapped_back_probabilities: Any,
        artifacts: Mapping[str, str | Path] | None = None,
    ) -> "ExperimentOutput":
        from .evaluation import evaluate_experiment_probabilities

        metrics = evaluate_experiment_probabilities(
            y_true,
            original_probabilities,
            swapped_back_probabilities,
        )
        return cls(metrics=metrics, artifacts=artifacts or {})


@dataclass(frozen=True)
class NotebookRunResult:
    run_id: str
    config_path: Path
    result_dir: Path
    artifact_dir: Path
    clearml_task_id: str | None
    validation_warnings: tuple[str, ...]
    leaderboard_updated: bool


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if start is not None and (current / "configs/project.json").is_file():
        return current
    for candidate in (current, *current.parents):
        if (candidate / "configs/project.json").is_file() and (
            candidate / "src/pmldl_llm"
        ).is_dir():
            return candidate
    raise ValueError(
        "Could not find the repository root. Pass project_root explicitly or "
        "run the notebook from inside the cloned repository."
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_experiment_config(
    setup: NotebookExperimentSetup,
    project: dict[str, Any],
) -> dict[str, Any]:
    allowed_roles = project.get("allowed_evaluation_roles", [])
    evaluation_role = setup.evaluation_role
    if evaluation_role is None:
        if len(allowed_roles) != 1:
            raise ValueError(
                "Set evaluation_role because the project currently allows "
                "multiple roles."
            )
        evaluation_role = allowed_roles[0]
    tracking_mode = setup.tracking_mode or project["tracking"]["default_mode"]
    config: dict[str, Any] = {
        "$schema": "../experiment.schema.json",
        "schema_version": project["schema_version"],
        "experiment_id": setup.experiment_id.strip(),
        "owner": setup.owner.strip(),
        "title": setup.title.strip(),
        "track": setup.track,
        "status": "planned",
        "hypothesis": setup.hypothesis.strip(),
        "parent_experiment_id": setup.parent_experiment_id,
        "changed_factor": setup.changed_factor.strip(),
        "seed": setup.seed,
        "evaluation_role": evaluation_role,
        "smoke_test": setup.smoke_test,
        "model": {
            "name": setup.model_name,
            "revision": setup.model_revision,
        },
        "training": dict(setup.training),
        "tracking": {
            "backend": project["tracking"]["backend"],
            "mode": tracking_mode,
        },
        "tags": list(setup.tags),
        "notes": setup.notes,
    }
    return config


def prepare_experiment_config(
    setup: NotebookExperimentSetup,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = find_project_root(project_root)
    project = _load_json_object(root / "configs/project.json")
    config = build_experiment_config(setup, project)
    validate_experiment_config(config, project)
    path = root / "configs/experiments" / f"{setup.experiment_id}.json"
    if path.exists():
        existing = _load_json_object(path)
        if existing != config:
            raise ValueError(
                f"{path} already exists with different values. Use a new "
                "experiment_id or edit and commit that config explicitly."
            )
        return path
    _atomic_json(path, config)
    return path


def run_notebook_experiment(
    experiment: Callable[[ExperimentRun], ExperimentOutput],
    setup: NotebookExperimentSetup,
    *,
    project_root: str | Path | None = None,
    update_leaderboard: bool = True,
) -> NotebookRunResult:
    """Run notebook code under every local, validation, and ClearML contract."""
    root = find_project_root(project_root)
    config_path = prepare_experiment_config(setup, project_root=root)
    seed_everything(setup.seed)

    with ExperimentRun(config_path, project_root=root) as run:
        output = experiment(run)
        if not isinstance(output, ExperimentOutput):
            raise TypeError(
                "The experiment function must return ExperimentOutput."
            )
        run.log_metrics(dict(output.metrics), namespace="validation")
        for artifact_name, artifact_path in output.artifacts.items():
            run.log_artifact(artifact_name, artifact_path)

    validation = validate_run_directory(run.result_dir, project_root=root)
    if not validation.valid:
        raise RuntimeError(
            "The generated run failed validation: " + " ".join(validation.errors)
        )

    leaderboard_updated = False
    warnings = list(validation.warnings)
    if update_leaderboard:
        collection = collect_result_rows(project_root=root)
        if collection.invalid_runs:
            warnings.append(
                "Leaderboard was not updated because another run is invalid."
            )
        else:
            project = _load_json_object(root / "configs/project.json")
            leaderboard = root / project["results"]["leaderboard"]
            write_leaderboard(leaderboard, collection)
            leaderboard_updated = True

    task_id = run.tracker.state.get("task_id")
    print(f"Completed run: {run.run_id}")
    print(f"Local result: {run.result_dir}")
    if task_id:
        print(f"ClearML task: {task_id}")
    if leaderboard_updated:
        print("Local leaderboard updated.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return NotebookRunResult(
        run_id=run.run_id,
        config_path=config_path,
        result_dir=run.result_dir,
        artifact_dir=run.artifact_dir,
        clearml_task_id=task_id,
        validation_warnings=tuple(warnings),
        leaderboard_updated=leaderboard_updated,
    )


def ask_experiment_setup(
    *,
    project_root: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
) -> NotebookExperimentSetup:
    """Collect the experiment contract interactively in a notebook cell."""
    root = find_project_root(project_root)
    project = _load_json_object(root / "configs/project.json")

    def required(prompt: str) -> str:
        while True:
            value = input_fn(prompt).strip()
            if value:
                return value
            print("This value is required.")

    def choice(prompt: str, choices: set[str], default: str) -> str:
        while True:
            value = input_fn(f"{prompt} [{default}]: ").strip() or default
            if value in choices:
                return value
            print("Choose one of: " + ", ".join(sorted(choices)))

    while True:
        experiment_id = required("Experiment ID, for example E030: ")
        if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
            break
        print("Use E followed by at least three digits.")

    owner = required("Owner name: ")
    title = required("Short experiment title: ")
    track = choice("Track", ALLOWED_TRACKS, "neural")
    hypothesis = required("Hypothesis: ")
    parent = input_fn("Parent experiment ID, blank if none: ").strip() or None
    changed_factor = required("Single changed factor: ")
    model_name = input_fn("Model name, blank if none: ").strip() or None
    revision = input_fn("Immutable model revision, blank if none: ").strip() or None
    default_seed = int(project["default_seed"])
    seed_text = input_fn(f"Seed [{default_seed}]: ").strip()
    seed = default_seed if not seed_text else int(seed_text)
    smoke_answer = input_fn("Smoke test? [Y/n]: ").strip().lower()
    smoke_test = smoke_answer not in {"n", "no"}
    tracking_mode = choice(
        "ClearML mode",
        {"online", "offline", "disabled"},
        str(project["tracking"]["default_mode"]),
    )
    training_text = input_fn("Training hyperparameters as JSON [{}]: ").strip()
    training = json.loads(training_text) if training_text else {}
    if not isinstance(training, dict):
        raise ValueError("Training hyperparameters must be a JSON object.")
    tags_text = input_fn("Comma-separated tags, blank if none: ").strip()
    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]

    return NotebookExperimentSetup(
        experiment_id=experiment_id,
        owner=owner,
        title=title,
        track=track,
        hypothesis=hypothesis,
        parent_experiment_id=parent,
        changed_factor=changed_factor,
        seed=seed,
        smoke_test=smoke_test,
        model_name=model_name,
        model_revision=revision,
        training=training,
        tags=tags,
        notes=input_fn("Notes, blank if none: ").strip(),
        tracking_mode=tracking_mode,
    )
