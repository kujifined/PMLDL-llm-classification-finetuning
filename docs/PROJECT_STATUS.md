# Project status and rubric gaps

This file deliberately separates completed, verified work from planned work.
The repository is a strong Phase 1 foundation, but it is not yet a final course
submission.

## Completed and verified

- Competition data contract, checksum audit, duplicate-prompt analysis, and
  null-turn handling.
- Immutable prompt-grouped split with dedicated selection, calibration, and
  untouched final-holdout roles.
- Uniform, class-prior, structural, sparse TF-IDF, and exploratory blend
  baselines with saved probabilities and reproducible metrics.
- A/B swap augmentation and exactly swap-symmetric inference.
- Balanced head-and-tail token-budget allocator for prompt/A/B with exhaustive
  small-case tests.
- Exact baseline dependency lock, per-run provenance hashes, and automated
  report-table consistency check.
- E2 on one H100: full fine-tuning, LoRA and QLoRA trained under one frozen
  split, seed and evaluation, with a four-candidate LoRA screening over rank,
  dropout and learning rate. At three epochs QLoRA reaches 1.03429 and LoRA
  1.04147, both ahead of the 1.0476 blend, while full fine-tuning degrades to
  1.08737. This is the first neural result that beats the classical baselines.

## Required before the course submission

1. Reproduce and attribute a public neural starter at an exact revision.
2. Make the fine-tuned cross-encoder beat the classical baselines. E2 trained
   one epoch and reached 1.0807 against 1.0476 for the blend, so the neural
   track needs more epochs, a learning-rate sweep for full fine-tuning, or both
   before it can be reported as the main model.
3. Run the controlled improvement ablation: same backbone/seed/compute, with
   balanced head-and-tail truncation and A/B swap versus the starter policy.
4. Refit the frozen winner on folds 0-7, calibrate on fold 8, then open fold 9
   exactly once for the final local estimate.
5. Build and verify the internet-off Kaggle inference notebook with attached
   model/tokenizer weights and a real hidden-test `submission.csv`.
6. Produce the anonymous `project.pdf` with architecture, experiments, plots,
   error analysis, limitations, contributions, and complete attribution.
7. Initialize or connect the team's private Git repository so provenance has a
   commit hash; run a clean-environment reproduction from that commit.

The course brief specifies teams of six and a hard Stage 2 deadline, but the
calendar date is not present in the supplied PDF. Record the Moodle deadline
before scheduling GPU experiments.

## Known environment deviations

The H100 runs cannot install the pinned locks: the internal mirror lacks those
exact versions, `transformers` is held at 4.56.2 because 4.57 cannot load a
4-bit model, ClearML is unavailable there, and the pinned checkpoint is
re-serialised to safetensors. Each deviation is described in
`infra/h100/README.md` and recorded in the run artifacts. The report must state
them rather than claim the locks were used.

## Next experiment (highest information value)

Run a paired neural experiment on fold 7:

- Arm A: exact, attributed reproduction of the selected public starter.
- Arm B: the same backbone, seed, maximum length, and step budget, changing only
  balanced head-and-tail truncation plus random A/B swap training and
  swap-averaged inference.

First benchmark 500 training steps. Continue to the full run only if projected
runtime leaves at least 1.5 hours under Kaggle's nine-hour notebook limit.
Compare log loss, grouped paired bootstrap delta, symmetry error,
truncation-stratified loss, and runtime. Freeze the winner before touching fold
8 or fold 9.
