# Results and artifacts

`results/` contains small, reviewable evidence that may be committed:

- `baselines/*/evaluation.json`: legacy multi-candidate baseline evaluations;
- `runs/<run_id>/config.json`: immutable configuration snapshot;
- `runs/<run_id>/run.json`: lifecycle, Git, ownership, and tracking metadata;
- `runs/<run_id>/metrics.json`: finite scalar summaries and optional history;
- `runs/<run_id>/study.json`: optional public-safe aggregate protocol and arm
  comparison, when a study emits one; no row-level predictions or checkpoints;
- `preflights/*.json`: aggregate memory or runtime feasibility evidence for
  protocols stopped before comparable evaluation metrics exist.

Every file named `metrics.json` follows `configs/metrics.schema.json` and
belongs to exactly one run. Multi-candidate baseline evaluations deliberately
use a different filename because they cannot satisfy that one-run contract.

`artifacts/` contains generated models, predictions, plots, tokenizer files,
and submissions. Those files can be large or licensed and are ignored by Git.
ClearML may hold a remote copy, but a ClearML outage never removes the local
run record.

Use `ExperimentRun` for new experiments. Do not write metrics into
`artifacts/`, and never put credentials or raw competition data into either
directory.

Run `python scripts/validate_run.py results/runs/<run_id>` before committing a
run. `python scripts/collect_results.py` validates all run directories and
atomically rebuilds `leaderboard.csv`; by default it includes only completed,
non-smoke runs from the currently allowed evaluation role.

Preflights are deliberately excluded from `leaderboard.csv`: they can justify
a resource decision, but they cannot rank model quality.
Completed runs whose immutable config sets `training.pilot.enabled=true` are
also excluded because their subsampled validation metrics are not comparable
with the frozen full-fold selection leaderboard.
