# Project status and remaining course gates

This page separates verified completion from the few external gates that are
not under the repository's control.

## Sprint 1: complete locally

- Competition data contract, checksum audit, duplicate-prompt analysis, and
  null-turn handling are implemented.
- The immutable prompt-grouped split keeps folds 0-6 for training, fold 7 for
  selection, fold 8 for calibration, and fold 9 for one final holdout.
- Uniform, class-prior, structural, sparse TF-IDF, and exploratory blend
  baselines have saved metrics and reconstruction checks.
- E2 compares full fine-tuning, LoRA, and QLoRA on `deberta-v3-base` with the
  same split, seed, maximum length, optimizer-step budget, A/B swap training,
  and swap-averaged evaluation. At three epochs, QLoRA is best at **1.03429**
  validation log loss, LoRA reaches 1.04147, and full fine-tuning degrades to
  1.08737.
- The final inference notebook is Internet-Off and test-size independent. Its
  clean Kaggle execution created a schema-valid `submission.csv`; the fixed
  58% QLoRA / 42% sparse blend received **1.02067** public log loss, ahead of
  QLoRA-only at 1.02832.
- `make test`, `make check-results`, `make collect-results`,
  `make verify-artifacts`, and `validate_run` for the one-, two-, and
  three-epoch E2 records passed before submission.

E6 is deliberately not included. It is an optional additional task; Sprint 1
priority is the required baselines and main experiments.

## Explicit limitations

- Kaggle's public score is useful external evidence, but it is not used to
  tune the blend or select another local model. Folds 8 and 9 remain unopened.
- The H100 environment used `transformers` 4.56.2 rather than the pinned
  4.57.6 and re-serialised the pinned model weights to safetensors. The run
  metadata and `infra/h100/README.md` record these deviations.
- The H100 mirror could not initialise ClearML (`tracking_status=unavailable`).
  The shared project now contains a dedicated **E2 historical import** task
  with the run identity, protocol, final metrics, and Kaggle result. The
  reproducible importer is `scripts/import_e2_to_clearml.py`; it uploads the
  complete 2,695-point history and `config.json`, `metrics.json`, and
  `run.json` into [the task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/77b645cca3cd46f08f2cdff46eb0c179/general).
  This record must remain labelled as a historical import, never as live
  tracking of the original H100 job.

## Remaining external actions

1. Run the historical importer with a valid personal ClearML API credential to
   attach the local history and JSON artifacts to the shared task. The task's
   summary and protocol are already visible in the project.
2. Obtain green pull-request CI after the team decides who opens the PR. The
   clean E2 branch is already published; `cherry-pick -x` trailers map the
   recorded run commits to the clean branch.
3. For Stage 2 only: refit the frozen winner on folds 0-7, calibrate on fold
   8, then evaluate fold 9 exactly once, and produce the course `project.pdf`.

`docs/SPRINT_1_E2_REPORT.md` is the concise submission-ready account of E2;
`docs/RESULTS.md` remains the detailed evidence ledger.
