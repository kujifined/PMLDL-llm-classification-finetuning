# Team notebook workflow

The ready-to-copy template is
`output/jupyter-notebook/team-managed-experiment.ipynb`. It is intended for a
cloned repository opened locally, in Colab, or in Kaggle.

Before the first run in a new environment, install the repository with the
tracking extra and configure personal ClearML credentials:

~~~bash
python -m pip install -e '.[tracking]'
clearml-init
~~~

Credentials remain outside Git. If ClearML is unavailable, the runner still
preserves the local run and marks the tracking state honestly.

The participant does only three things:

1. Runs the questionnaire cell and enters the assigned experiment ID, owner,
   hypothesis, one changed factor, model, seed, and training parameters.
2. Places existing training/evaluation code inside `train_and_evaluate` and
   returns `ExperimentOutput.from_predictions(...)` with validation labels,
   ordinary predictions, and A/B-swapped predictions mapped back to A/B/tie.
3. Runs the notebook top to bottom.

The runner automatically creates or reuses the immutable experiment config,
sets Python/NumPy/PyTorch seeds, opens exactly one `ExperimentRun`, connects one
ClearML task, persists failures, logs final metrics and artifacts, validates the
three local run files, and rebuilds the leaderboard when all runs are valid.

## Required output

A full experiment must return these validation metrics:

- `log_loss`;
- `accuracy`;
- `macro_f1`;
- `ece_15`;
- `brier_score`;
- `swap_error_l1`.

`runtime_seconds` is measured automatically. Artifact paths are optional and
may point to models, validation predictions, plots, or tokenizer files.
`ExperimentOutput.from_predictions` automatically applies A/B symmetry
averaging and calculates all six quality metrics; the participant does not
need to implement those formulas independently.
The multiclass Brier score is the row-mean sum of squared one-hot errors, and
`swap_error_l1` is the row-mean L1 distance before symmetry averaging.

## Smoke and full runs

Use `smoke_test=true` first; it may report only the metrics already available
while code and data flow are being checked. A full run requires the complete
metric list, a committed config, and a clean Git checkout. This makes the Git
revision stored in `run.json` sufficient to reproduce the exact code and
configuration used by the team member.

If the same experiment ID already has a different config, the runner stops
instead of overwriting it. The participant must either use a new assigned ID or
explicitly edit and commit the existing config.
