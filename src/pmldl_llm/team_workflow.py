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
    ".kaggle/",
    ".clearml/",
    "clearml_offline_session/",
    "data/llm-classification-finetuning/",
    "checkpoints/",
)
FORBIDDEN_GIT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)
FORBIDDEN_GIT_NAMES = {
    ".env",
    "clearml.conf",
    "credentials.json",
    "kaggle.json",
    "secrets.json",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?m)^\s*(?:CLEARML_API_(?:ACCESS|SECRET)_KEY|KAGGLE_KEY)\s*=\s*\S+"
    ),
)


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


def _stage_and_commit(root: Path, message: str, *paths: Path) -> str:
    relative_paths = [path.resolve().relative_to(root).as_posix() for path in paths]
    _git(root, "add", "--", *relative_paths)
    staged = _git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    unexpected = [
        path
        for path in staged
        if not any(
            path == allowed or path.startswith(allowed.rstrip("/") + "/")
            for allowed in relative_paths
        )
    ]
    if unexpected:
        raise RuntimeError(
            "Refusing to include files staged outside this experiment: "
            + ", ".join(unexpected)
        )
    forbidden = [
        path
        for path in staged
        if path.startswith(FORBIDDEN_GIT_PREFIXES)
        or path.endswith(FORBIDDEN_GIT_SUFFIXES)
        or Path(path).name in FORBIDDEN_GIT_NAMES
        or Path(path).name.startswith(".env.")
    ]
    if forbidden:
        raise RuntimeError("Refusing to commit private or large files: " + ", ".join(forbidden))
    secret_hits: list[str] = []
    for path in staged:
        indexed = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if indexed.returncode != 0:
            continue
        if b"\0" in indexed.stdout[:8192]:
            continue
        text = indexed.stdout.decode("utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secret_hits.append(path)
    if secret_hits:
        raise RuntimeError(
            "Refusing to commit files containing credential-like values: "
            + ", ".join(secret_hits)
        )
    if not staged:
        raise RuntimeError("There are no changes to commit.")
    _git(root, "commit", "-m", message)
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    remaining = _git(root, "diff", "--cached", "--name-only").stdout.strip()
    if remaining:
        raise RuntimeError(
            f"Commit {commit[:8]} was created, but the Git index is not clean: {remaining}"
        )
    return commit


def prepare_full_run(root: Path, experiment_id: str) -> str:
    root = root.resolve()
    config_path = root / "configs/experiments" / f"{experiment_id}.json"
    original_config_bytes = config_path.read_bytes()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke_candidates = [
        (directory, run)
        for directory, run in _matching_runs(root, experiment_id)
        if run["status"] == "completed" and run["smoke_test"]
    ]
    if not smoke_candidates:
        raise RuntimeError("Run the generated notebook once as a smoke test first.")
    smoke_run_dir, _smoke_run = max(
        smoke_candidates, key=lambda item: item[1]["ended_at"]
    )
    config["smoke_test"] = False
    project = json.loads((root / "configs/project.json").read_text(encoding="utf-8"))
    validate_experiment_config(config, project)
    _atomic_json(config_path, config)
    notebook_path = root / config["training"]["notebook"]
    original_notebook_bytes = notebook_path.read_bytes()
    clean_notebook_outputs(notebook_path)

    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        config_path.write_bytes(original_config_bytes)
        notebook_path.write_bytes(original_notebook_bytes)
        raise RuntimeError("Tests failed; full run was not prepared.")

    return _stage_and_commit(
        root,
        f"Prepare {experiment_id} full experiment",
        config_path,
        notebook_path,
        smoke_run_dir,
    )


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
        config_path,
        root / config["training"]["notebook"],
        run_dir,
        leaderboard,
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
