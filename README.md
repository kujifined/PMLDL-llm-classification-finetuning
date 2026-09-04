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

## Quick start

Use Python 3.12 for exact baseline reproduction. The lock file records the
tested environment; pyproject.toml provides looser compatibility bounds for
development.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.lock
python -m pip install -e . --no-deps

make test
make audit
make baseline
make sparse-baseline
make blend-baselines
make check-results
make verify-artifacts
~~~

The canonical split is already frozen under data/splits/. Do not regenerate it
during model development. `scripts/freeze_folds.py` refuses to overwrite it
unless `--force` is passed explicitly.

Generated predictions and checkpoints are written under artifacts/ and are not
committed. Small aggregate `metrics.json` files are intentionally versionable
as evidence. Files named `test_schema_predictions.csv` contain only the local
three-row demonstration test and are never Kaggle submissions.

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
- Reported metrics must be loaded from saved artifacts; `make check-results`
  rejects drift between the report table and those artifacts.
- `make verify-artifacts` reconstructs every structural, sparse, and blend
  validation probability from the named checkpoints and fails on any mismatch.
- Every run verifies the raw-data checksums and records data, fold, config, and
  code hashes in its metrics.

## Project layout

~~~text
configs/        versioned experiment configuration
data/           local raw data, checksums, and data documentation
docs/           experiment plan, decisions, and source attribution
scripts/        reproducible command-line entry points
src/pmldl_llm/  reusable data, split, feature, metric, and submission code
tests/          executable invariants
artifacts/      generated metrics, predictions, models, and submissions
~~~

See docs/PROJECT_STATUS.md for the honest rubric gap analysis and
docs/EXPERIMENT_PLAN.md for the planned model ladder and stopping rules. The
course-aligned packaging gates are in docs/SUBMISSION_CHECKLIST.md.
