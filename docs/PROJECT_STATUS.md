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
- ClearML was unavailable on the H100 mirror (`tracking_status=unavailable`).
  Local provenance, metrics, predictions, checkpoints, and validation gates
  remain intact. This must be disclosed until a task is visible in the shared
  ClearML project; it must not be presented as live tracking retroactively.

## Remaining external actions

1. Resolve shared ClearML-project access or obtain course-team acceptance of
   the recorded local evidence as the temporary substitute.
2. Push the clean E2 branch and obtain green pull-request CI once repository
   network access and ownership are resolved. The branch has not been altered
   to hide provenance; `cherry-pick -x` trailers map the recorded run commits
   to the clean branch.
3. For Stage 2 only: refit the frozen winner on folds 0-7, calibrate on fold
   8, then evaluate fold 9 exactly once, and produce the course `project.pdf`.

`docs/SPRINT_1_E2_REPORT.md` is the concise submission-ready account of E2;
`docs/RESULTS.md` remains the detailed evidence ledger.
