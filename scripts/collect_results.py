#!/usr/bin/env python3
"""Build a deterministic leaderboard from validated ExperimentRun records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pmldl_llm.result_collection import collect_result_rows, write_leaderboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Project config must contain an object.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.json"),
    )
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-role",
        choices=("selection", "calibration", "final_holdout"),
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include completed smoke tests; excluded by default.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Reject runs whose ignored artifact directory is unavailable.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Write a partial leaderboard while reporting invalid run directories.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    project_path = args.project_config
    if not project_path.is_absolute():
        project_path = root / project_path
    try:
        project = load_project(project_path)
        collection = collect_result_rows(
            project_root=root,
            project_config_path=project_path,
            runs_dir=args.runs_dir,
            evaluation_role=args.evaluation_role,
            include_smoke=args.include_smoke,
            require_artifacts=args.require_artifacts,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for invalid in collection.invalid_runs:
        print(f"INVALID: {invalid.run_dir}", file=sys.stderr)
        for error in invalid.errors:
            print(f"  - {error}", file=sys.stderr)
    if collection.invalid_runs and not args.skip_invalid:
        print(
            "ERROR: leaderboard was not changed; fix invalid runs or use "
            "--skip-invalid.",
            file=sys.stderr,
        )
        return 1

    output = args.output
    if output is None:
        output = root / project["results"]["leaderboard"]
    elif not output.is_absolute():
        output = root / output
    write_leaderboard(output, collection)
    excluded = sum(collection.excluded_counts.values())
    print(
        f"OK: wrote {len(collection.rows)} comparable runs to {output}; "
        f"excluded {excluded}, invalid {len(collection.invalid_runs)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
