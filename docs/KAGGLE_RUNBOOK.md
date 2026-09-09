# Kaggle execution runbook

The managed experiment notebook is the source of truth. Kaggle is only the GPU
executor; results must be downloaded and validated in this repository.

## Before upload

1. Finish a successful local or Kaggle smoke run.
2. Run `make prepare-full EXPERIMENT=<ID> PYTHON=.venv/bin/python`.
3. Confirm that `git status --short` is empty and that the intended full-run
   commit contains no data, tokens, `.env`, checkpoints, or ClearML config.
4. Build `artifacts/kaggle/repository.bundle` from the clean commit with
   `scripts/build_kaggle_bundle.py`.
5. Upload the bundle as a private Kaggle dataset only after explicit approval.

## Kaggle inputs

- official `llm-classification-finetuning` competition data;
- private repository bundle;
- private wheel bundle matching `requirements-transformer.lock` when Internet
  is disabled;
- attached pinned base model and selected adapter for offline inference;
- ClearML credentials only for training notebooks and only after explicit
  approval. They are never required by the final Internet-Off inference notebook.

## Training notebook

Use `output/kaggle/run_managed_experiment.ipynb`. Configure the bundle path and
managed notebook path in its first cell, select one T4 for E2, and run all cells.
The launcher clones the tracked commit into `/kaggle/working`, installs pinned
dependencies, and executes the generated notebook with the repository root as
its working directory.

Download the complete Kaggle working output. Copy only the generated
`results/runs/<run_id>` and `artifacts/<run_id>` directories back to the same
clean commit, then run all local validation gates.

## Submission boundary

Creating a valid `submission.csv` is not authorization to submit it. Before any
Kaggle submission, report the selected local metric, inference runtime, schema
validation, and exact notebook/model revisions, then request explicit approval.
