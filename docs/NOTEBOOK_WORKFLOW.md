# Self-service notebook workflow

No experiment ID, branch name, validation fold, parent, seed, or ClearML task
needs to be requested from the Team Lead. From a clean `main` checkout run:

~~~bash
make new-experiment
~~~

The wizard asks only four short questions: experiment name, hypothesis, the
single changed factor, and model name (optionally `model@revision`). It then:

- generates a globally unique numeric experiment ID;
- reads the owner from `git config user.name`;
- selects the current leaderboard leader as parent;
- uses the project seed and active evaluation role;
- creates `experiment/<ID>-<slug>`;
- writes the validated config;
- creates a personal notebook from the managed template;
- enables ClearML online tracking and smoke mode.

The participant only implements `train_and_evaluate` in the generated notebook
and returns `ExperimentOutput.from_predictions(...)`. The helper calculates
log loss, accuracy, macro F1, ECE-15, Brier score, and A/B swap error. Runtime is
measured automatically.

After implementing `train_and_evaluate`, execute the complete two-stage workflow
with one command:

~~~bash
make run-experiment EXPERIMENT=<ID>
~~~

The command runs the notebook in smoke mode first. If the smoke run fails, it
stops with the failing notebook cell and keeps the config in smoke mode so the
participant can fix the code and retry. If smoke succeeds, it runs repository
checks, commits the clean full-run revision, reloads the notebook in full mode,
and executes it. Re-running after a failed full run retries only the full phase.

After the successful full run execute:

~~~bash
make submit-experiment EXPERIMENT=<ID>
~~~

This validates the full run, rebuilds `results/leaderboard.csv`, commits the
versioned results, pushes the experiment branch, and prints the GitHub compare
link. The participant opens and reviews the pull request manually. Raw Kaggle
data, tokens, checkpoints, and `artifacts/` remain excluded by Git.

If ClearML or GitHub authentication has not been configured on a machine, the
corresponding official login is still a one-time personal setup. A ClearML
failure never destroys the local run.
