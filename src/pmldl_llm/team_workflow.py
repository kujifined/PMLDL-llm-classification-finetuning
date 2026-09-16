from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment import validate_experiment_config
from .notebook import (
    NotebookExperimentSetup,
    ask_experiment_setup,
    clean_notebook_outputs,
    prepare_experiment_config,
)
from .result_collection import collect_result_rows, write_leaderboard
from .run_validation import validate_run_directory


FORBIDDEN_GIT_PREFIXES = (
    "artifacts/",
    "data/llm-classification-finetuning/",
)
FORBIDDEN_GIT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx")


@dataclass(frozen=True)
class ScaffoldedExperiment:
    experiment_id: str
    branch: str
    config_path: Path
    notebook_path: Path


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed


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


def slugify(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    return "-".join(words[:6])[:48] or "experiment"


def _write_generated_notebook(
    *,
    root: Path,
    setup: NotebookExperimentSetup,
    config_path: Path,
) -> Path:
    template_path = root / "output/jupyter-notebook/team-managed-experiment.ipynb"
    notebook = json.loads(template_path.read_text(encoding="utf-8"))
    relative_config = config_path.relative_to(root).as_posix()
    serialized = json.dumps(notebook, ensure_ascii=False)
    serialized = serialized.replace("__EXPERIMENT_CONFIG__", relative_config)
    notebook = json.loads(serialized)
    notebook["cells"][0]["source"][0] = (
        f"# {setup.experiment_id}: {setup.title}\n"
    )
    notebook_path = (
        root
        / "output/jupyter-notebook"
        / f"{setup.experiment_id}__{slugify(setup.title)}.ipynb"
    )
    _atomic_json(notebook_path, notebook)
    return notebook_path


def scaffold_experiment(
    *,
    root: Path,
    setup: NotebookExperimentSetup,
    create_branch: bool = True,
) -> ScaffoldedExperiment:
    root = root.resolve()
    if create_branch:
        if _git(root, "status", "--porcelain").stdout.strip():
            raise RuntimeError("Commit or stash current changes before starting an experiment.")
        branch = f"experiment/{setup.experiment_id}-{slugify(setup.title)}"
        _git(root, "switch", "-c", branch)
    else:
        branch = f"experiment/{setup.experiment_id}-{slugify(setup.title)}"

    config_path = prepare_experiment_config(setup, project_root=root)
    notebook_path = _write_generated_notebook(
        root=root,
        setup=setup,
        config_path=config_path,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["notebook"] = notebook_path.relative_to(root).as_posix()
    project = json.loads((root / "configs/project.json").read_text(encoding="utf-8"))
    validate_experiment_config(config, project)
    _atomic_json(config_path, config)
    return ScaffoldedExperiment(
        experiment_id=setup.experiment_id,
        branch=branch,
        config_path=config_path,
        notebook_path=notebook_path,
    )


def create_interactive_experiment(root: Path) -> ScaffoldedExperiment:
    setup = ask_experiment_setup(project_root=root)
    return scaffold_experiment(root=root, setup=setup)


def _matching_runs(root: Path, experiment_id: str) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    run_root = root / "results/runs"
    if not run_root.is_dir():
        return matches
    for directory in sorted(run_root.iterdir()):
        if not directory.is_dir():
            continue
        report = validate_run_directory(directory, project_root=root)
        if not report.valid or report.run is None:
            continue
        if report.run["experiment_id"] == experiment_id:
            matches.append((directory, report.run))
    return matches


def _execute_notebook(root: Path, notebook_path: Path, *, phase: str) -> str:
    """Execute a notebook in place and return its path.

    Notebook execution is intentionally kept here, next to the team workflow,
    so a participant does not have to copy/paste a second command.  The
    executed notebook is written even on failure: its traceback is useful when
    fixing a smoke-test error locally.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError as exc:
        raise RuntimeError(
            "Notebook runner is not installed. Install it once with "
            "python -m pip install nbclient nbformat."
        ) from exc

    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except (OSError, nbformat.reader.NotJSONError, ValueError) as exc:
        raise RuntimeError(f"Cannot read experiment notebook {notebook_path}: {exc}") from exc

    timeout = 24 * 60 * 60
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except Exception as exc:
        # Persist cell outputs and the traceback before presenting a short,
        # actionable error to the participant.
        nbformat.write(notebook, notebook_path)
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 500:
            detail = detail[:497] + "..."
        if isinstance(exc, CellExecutionError):
            action = "Fix the first failing cell"
        else:
            action = "Check the Jupyter kernel and installed dependencies"
        raise RuntimeError(
            f"{phase} notebook run failed. {action} in "
            f"{notebook_path.relative_to(root)}, then run the command again. "
            f"Details: {detail or type(exc).__name__}"
        ) from exc

    nbformat.write(notebook, notebook_path)
    return notebook_path.as_posix()


def run_experiment(root: Path, experiment_id: str) -> dict[str, str]:
    """Run the self-service notebook through smoke and full phases.

    A freshly scaffolded experiment starts in smoke mode.  Once that run is
    completed, the existing ``prepare_full_run`` transition commits a clean
    full-run revision.  Re-running this command after a failed full run skips
    smoke and retries only the full phase.
    """
    root = root.resolve()
    if not experiment_id or not re.fullmatch(r"E[0-9]{3,}", experiment_id):
        raise RuntimeError(
            "Provide a valid experiment ID, for example: "
            "make run-experiment EXPERIMENT=E20260905000000000001"
        )
    config_path = root / "configs/experiments" / f"{experiment_id}.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Experiment config was not found: {config_path.relative_to(root)}. "
            "Create it first with make new-experiment."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        notebook_value = config["training"]["notebook"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Cannot load notebook path from {config_path}: {exc}") from exc
    notebook_path = (root / str(notebook_value)).resolve()
    try:
        notebook_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("The experiment notebook must stay inside the repository.") from exc
    if not notebook_path.is_file():
        raise RuntimeError(f"Experiment notebook was not found: {notebook_path.relative_to(root)}")

    smoke_runs = [
        (directory, run)
        for directory, run in _matching_runs(root, experiment_id)
        if run["smoke_test"] and run["status"] == "completed"
    ]
    if config.get("smoke_test", True):
        _execute_notebook(root, notebook_path, phase="Smoke")
        smoke_runs = [
            (directory, run)
            for directory, run in _matching_runs(root, experiment_id)
            if run["smoke_test"] and run["status"] == "completed"
        ]
        if not smoke_runs:
            raise RuntimeError(
                "Smoke notebook finished without a valid completed run. "
                "Inspect results/runs/ and fix the experiment before retrying."
            )
        prepare_full_run(root, experiment_id)

    _execute_notebook(root, notebook_path, phase="Full")
    full_runs = [
        (directory, run)
        for directory, run in _matching_runs(root, experiment_id)
        if not run["smoke_test"] and run["status"] == "completed"
    ]
    if not full_runs:
        raise RuntimeError(
            "Full notebook finished without a valid completed run. "
            "Inspect results/runs/ for the validation error and retry."
        )
    smoke_candidate_runs = smoke_runs or [
        (directory, run)
        for directory, run in _matching_runs(root, experiment_id)
        if run["smoke_test"]
    ]
    if not smoke_candidate_runs:
        raise RuntimeError(
            "No completed smoke run was found for this experiment. "
            "Restore smoke mode or create a new experiment with make new-experiment."
        )
    return {
        "smoke_run_id": max(
            smoke_candidate_runs, key=lambda item: item[1]["ended_at"]
        )[1]["run_id"],
        "full_run_id": max(
            full_runs, key=lambda item: item[1]["ended_at"]
        )[1]["run_id"],
    }


def _stage_and_commit(root: Path, message: str, *paths: Path) -> str:
    relative_paths = [path.resolve().relative_to(root).as_posix() for path in paths]
    _git(root, "add", "--", *relative_paths)
    staged = _git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    forbidden = [
        path
        for path in staged
        if path.startswith(FORBIDDEN_GIT_PREFIXES)
        or path.endswith(FORBIDDEN_GIT_SUFFIXES)
        or Path(path).name == "kaggle.json"
        or Path(path).name == ".env"
        or Path(path).name.startswith(".env.")
    ]
    if forbidden:
        raise RuntimeError("Refusing to commit private or large files: " + ", ".join(forbidden))
    if not staged:
        raise RuntimeError("There are no changes to commit.")
    _git(root, "commit", "-m", message)
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    remaining = _git(root, "status", "--porcelain").stdout.strip()
    if remaining:
        raise RuntimeError(
            f"Commit {commit[:8]} was created, but the worktree is not clean: {remaining}"
        )
    return commit


def prepare_full_run(root: Path, experiment_id: str) -> str:
    root = root.resolve()
    config_path = root / "configs/experiments" / f"{experiment_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not any(
        run["status"] == "completed" and run["smoke_test"]
        for _directory, run in _matching_runs(root, experiment_id)
    ):
        raise RuntimeError("Run the generated notebook once as a smoke test first.")
    config["smoke_test"] = False
    project = json.loads((root / "configs/project.json").read_text(encoding="utf-8"))
    validate_experiment_config(config, project)
    _atomic_json(config_path, config)
    notebook_path = root / config["training"]["notebook"]
    clean_notebook_outputs(notebook_path)

    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        # Keep the experiment retryable: a failed repository check must not
        # strand the config in full mode without the clean commit it requires.
        config["smoke_test"] = True
        _atomic_json(config_path, config)
        raise RuntimeError("Tests failed; full run was not prepared.")

    return _stage_and_commit(root, f"Prepare {experiment_id} full experiment", root)


def submit_experiment(
    root: Path,
    experiment_id: str,
    *,
    push: bool = True,
) -> tuple[str, str | None]:
    root = root.resolve()
    candidates = [
        (directory, run)
        for directory, run in _matching_runs(root, experiment_id)
        if run["status"] == "completed" and not run["smoke_test"]
    ]
    if not candidates:
        raise RuntimeError("No completed full run exists for this experiment.")
    run_dir, _run = max(candidates, key=lambda item: item[1]["ended_at"])

    project = json.loads((root / "configs/project.json").read_text(encoding="utf-8"))
    config_path = root / "configs/experiments" / f"{experiment_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    clean_notebook_outputs(root / config["training"]["notebook"])
    collection = collect_result_rows(project_root=root)
    if collection.invalid_runs:
        raise RuntimeError("At least one run is invalid; leaderboard was not changed.")
    leaderboard = root / project["results"]["leaderboard"]
    write_leaderboard(leaderboard, collection)
    commit = _stage_and_commit(
        root,
        f"Record {experiment_id} full experiment",
        root,
    )

    compare_url = None
    if push:
        _git(root, "push", "-u", "origin", "HEAD")
        branch = _git(root, "branch", "--show-current").stdout.strip()
        remote = _git(root, "remote", "get-url", "origin").stdout.strip()
        if remote.startswith("git@github.com:"):
            remote = "https://github.com/" + remote.removeprefix("git@github.com:")
        if remote.startswith("https://github.com/"):
            compare_url = remote.removesuffix(".git") + f"/compare/main...{branch}?expand=1"
    return commit, compare_url
