#!/usr/bin/env python3
"""Build a private Kaggle Git bundle from the current clean tracked commit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "kaggle" / "repository.bundle"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if git("status", "--porcelain"):
        raise RuntimeError("Refusing to bundle a dirty worktree.")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if not branch.startswith("experiment/"):
        raise RuntimeError("Kaggle bundles must be built from an experiment branch.")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "bundle", "create", str(output), "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "bundle", "verify", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"Created {output}")
    print(f"Commit: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
