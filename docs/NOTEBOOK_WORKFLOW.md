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

Run the notebook once for a smoke test. Then execute:

~~~bash
make prepare-full EXPERIMENT=<ID>
~~~

This command verifies that a completed smoke run exists, changes the config to
full mode, runs the tests, and commits the notebook, config, code, and smoke
evidence. Restart the notebook kernel so it reloads the full config, then run
the notebook again.

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
