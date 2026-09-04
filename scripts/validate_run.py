#!/usr/bin/env python3
"""Validate one ExperimentRun directory and its cross-file invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pmldl_llm.run_validation import validate_run_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to results/runs/<run_id>.")
    parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.json"),
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail when the ignored artifacts/<run_id> directory is unavailable.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable report."
    )
    args = parser.parse_args()

    try:
        report = validate_run_directory(
            args.run_dir,
            project_root=args.project_root,
            project_config_path=args.project_config,
            require_artifacts=args.require_artifacts,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        label = report.run.get("run_id") if report.run else report.run_dir.name
        print(f"{'OK' if report.valid else 'INVALID'}: {label}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
