#!/usr/bin/env python3
"""Self-service experiment workflow for team members."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.team_workflow import (
    create_interactive_experiment,
    prepare_full_run,
    submit_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="Create ID, branch, config, and notebook.")
    prepare = subparsers.add_parser("prepare-full", help="Commit a clean full run.")
    prepare.add_argument("experiment_id")
    submit = subparsers.add_parser(
        "submit", help="Validate, commit, push, and print the PR link."
    )
    submit.add_argument("experiment_id")
    submit.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "start":
            result = create_interactive_experiment(PROJECT_ROOT)
            print(f"\nCreated {result.experiment_id}")
            print(f"Branch: {result.branch}")
            print(f"Notebook: {result.notebook_path.relative_to(PROJECT_ROOT)}")
            print("Open the notebook, implement train_and_evaluate, and Run All.")
        elif args.command == "prepare-full":
            commit = prepare_full_run(PROJECT_ROOT, args.experiment_id)
            print(f"Full run is ready at clean commit {commit[:8]}.")
            print("Restart the notebook kernel and Run All again.")
        else:
            commit, compare_url = submit_experiment(
                PROJECT_ROOT,
                args.experiment_id,
                push=not args.no_push,
            )
            print(f"Recorded result at commit {commit[:8]}.")
            if compare_url:
                print("Create the pull request yourself:")
                print(compare_url)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
