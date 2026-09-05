# LLM Classification Fine-tuning

Course project for Practical Machine Learning and Deep Learning. The task is
three-class preference prediction for Chatbot Arena conversations:
model A wins, model B wins, or tie.

The repository is designed around two goals:

1. a competitive Kaggle solution;
2. a defensible, reproducible experiment that satisfies the course rubric.

Kaggle competition: [LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning/data)

## Current status

- Raw competition files are present locally and protected by .gitignore.
- Data-contract checks handle multi-turn JSON and null response turns.
- A checked, immutable id-to-fold assignment keeps repeated normalized prompts
  in one validation group across every experiment.
- A reproducible structural baseline is implemented.
- A leakage-safe sparse TF-IDF baseline is implemented.
- A/B swap augmentation and swap-averaged inference are enforced.
- Tokenizer-independent balanced head-and-tail truncation is implemented and
  covered by property-style tests for the neural stage.
- Transformer fine-tuning is the next implementation milestone.

The local test.csv has only three demonstration rows. Kaggle replaces it with
the hidden scoring set, so all inference code must be independent of test size.

## Team experiment contract

Team-wide settings live in `configs/project.json`. During the current model
selection stage, only the selection fold is allowed for experiment comparison.
The Team Lead changes this policy before calibration or final-holdout access.

For a new hypothesis, run `make new-experiment`. A four-question wizard creates
the unique experiment ID, branch, schema-valid config, parent link, defaults,
and personal notebook automatically. Do not edit the shared template or create
a new validation split.

All runs will be tracked under the shared ClearML project
`PMLDL LLM Classification Finetuning`. ClearML credentials remain local and
must never be committed.

For notebook experiments, the participant edits only `train_and_evaluate` in
the generated notebook. The shared runner handles config, seeding, ClearML,
validation, artifacts, and leaderboard updates. See `docs/NOTEBOOK_WORKFLOW.md`.

## Quick start

Use Python 3.12 for exact baseline reproduction. The lock file records the
tested environment; pyproject.toml provides looser compatibility bounds for
development.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.lock
python -m pip install -e . --no-deps
# one-time dependencies for ClearML and `make run-experiment`:
python -m pip install 'clearml>=1.17,<3' 'nbclient>=0.8,<1' 'nbformat>=5.9,<6'

make test
make audit
make baseline
make baseline-run
make sparse-baseline
make blend-baselines
make check-results
make collect-results
make verify-artifacts
~~~

## Self-service experiment

~~~bash
make new-experiment
# answer four short questions and edit train_and_evaluate in the generated notebook
make run-experiment EXPERIMENT=<ID>
make submit-experiment EXPERIMENT=<ID>
~~~

The run command executes a smoke test and, only if it passes, the full notebook
automatically. The last command validates, records, pushes, and prints the
pull-request link.
The participant creates the PR manually. No ID, branch name, seed, fold, parent
experiment, or approval is requested from the Team Lead.

The canonical split is already frozen under data/splits/. Do not regenerate it
during model development. `scripts/freeze_folds.py` refuses to overwrite it
unless `--force` is passed explicitly.

Generated predictions and checkpoints are written under `artifacts/` and are
not committed. Comparable metrics and run metadata are written under `results/`
and are intentionally versionable evidence. Files named
`test_schema_predictions.csv` contain only the local
three-row demonstration test and are never Kaggle submissions.

Every new run writes one schema-controlled `metrics.json`. Validate a run and
rebuild the team leaderboard with:

~~~bash
python scripts/validate_run.py results/runs/<run_id>
make collect-results
~~~

The collector includes only valid, completed, non-smoke runs from one
evaluation role and sorts them by the primary metric from `configs/project.json`.

`make baseline-run` is the end-to-end contract check for the existing E002
structural baseline. It trains the real model, writes the canonical run under
`results/runs/`, keeps large files under ignored `artifacts/`, and can then be
added to the leaderboard with `make collect-results`. It requires the Kaggle
files described in `data/README.md` and a clean Git commit.

Every pull request runs the data-independent repository checks in GitHub
Actions. Baseline training, data audit, and artifact reconstruction remain local
checks because licensed Kaggle data and model artifacts are never uploaded to CI.

## Validation contract

The split was created once with a deterministic 10-fold StratifiedGroupKFold
keyed by a hash of the normalized full prompt. All training and evaluation
scripts load and verify the same `data/splits/folds.csv`; they never recompute
folds dynamically. Roles were fixed before model development:

- folds 0-6: training;
- fold 7: model selection and ablations;
- fold 8: probability calibration;
- fold 9: final untouched holdout.

The final holdout must not be used to choose architecture, hyperparameters,
truncation, seed, or ensemble weights. Kaggle leaderboard scores are external
checks, not a replacement for local validation.

## Key invariants

- Target order is always A, B, tie.
- model_a and model_b identities are never model inputs because Kaggle test
  does not provide them.
- JSON null responses are handled explicitly.
- Prompt groups never cross validation boundaries.
- Swapping A and B swaps the corresponding labels and probabilities.
- Every submission contains finite non-negative probabilities summing to one.
- Reported metrics must be loaded from saved results; `make check-results`
  rejects drift between the report table and those JSON files.
- `make verify-artifacts` reconstructs every structural, sparse, and blend
  validation probability from the named checkpoints and fails on any mismatch.
- Every run verifies the raw-data checksums and records data, fold, config, and
  code hashes in its metrics.

## Project layout

~~~text
configs/        versioned experiment configuration
data/           local raw data, checksums, and data documentation
docs/           experiment plan, decisions, and source attribution
output/         reusable managed notebook template
scripts/        reproducible command-line entry points
src/pmldl_llm/  reusable data, split, feature, metric, and submission code
tests/          executable invariants
results/        versioned metrics, run metadata, and comparison tables
artifacts/      ignored predictions, models, plots, and submissions
~~~

See docs/PROJECT_STATUS.md for the honest rubric gap analysis and
docs/EXPERIMENT_PLAN.md for the planned model ladder and stopping rules. The
course-aligned packaging gates are in docs/SUBMISSION_CHECKLIST.md.
