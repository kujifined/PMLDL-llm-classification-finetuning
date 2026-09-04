# Results and artifacts

`results/` contains small, reviewable evidence that may be committed:

- `baselines/`: canonical aggregate metrics used by the report;
- `runs/<run_id>/config.json`: immutable configuration snapshot;
- `runs/<run_id>/run.json`: lifecycle, Git, ownership, and tracking metadata;
- `runs/<run_id>/metrics.json`: finite scalar summaries and optional history.

`artifacts/` contains generated models, predictions, plots, tokenizer files,
and submissions. Those files can be large or licensed and are ignored by Git.
ClearML may hold a remote copy, but a ClearML outage never removes the local
run record.

Use `ExperimentRun` for new experiments. Do not write metrics into
`artifacts/`, and never put credentials or raw competition data into either
directory.
