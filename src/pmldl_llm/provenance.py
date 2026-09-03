from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from .data import file_sha256


def combined_sha256(paths: Iterable[str | Path], root: str | Path) -> str:
    """Hash file names and bytes in a deterministic order."""
    root_path = Path(root).resolve()
    resolved_paths = sorted({Path(path).resolve() for path in paths})
    digest = hashlib.sha256()
    for path in resolved_paths:
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError:
            relative = f"external:{path.as_posix()}"
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def path_label(path: str | Path, project_root: str | Path) -> str:
    """Return a project-relative label, or an explicit absolute external path."""
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def git_state(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_provenance(
    project_root: str | Path,
    script_path: str | Path,
    config_paths: Iterable[str | Path],
    fold_path: str | Path,
    dataset_hashes: dict[str, str],
) -> dict[str, object]:
    root = Path(project_root).resolve()
    configs = [Path(path).resolve() for path in config_paths]
    code_files = sorted((root / "src" / "pmldl_llm").glob("*.py")) + [
        Path(script_path).resolve()
    ]
    return {
        "git": git_state(root),
        "code_sha256": combined_sha256(code_files, root),
        "code_files": [path_label(path, root) for path in code_files],
        "configs": {
            path_label(path, root): file_sha256(path) for path in configs
        },
        "folds": {
            "path": path_label(fold_path, root),
            "sha256": file_sha256(fold_path),
        },
        "dataset_sha256": dict(sorted(dataset_hashes.items())),
    }
