# Gemma-2 9B Kaggle pilot

## Decision

The original Gemma-2 9B experiment remains closed as resource-limited on two
Tesla T4 GPUs. This is a new, explicitly non-comparable pilot approved after
the team confirmed that no larger compute environment is available.

## Frozen pilot protocol

- Model: the same pinned official Gemma-2-9B-IT revision.
- Arms: BF16/FP16 LoRA and NF4 QLoRA with the same adapter configuration.
- Sampling: deterministic and class-balanced inside the existing folds.
- Training: 300 rows per class from folds 0-6, 900 rows total.
- Evaluation: 60 rows per class from fold 7, 180 rows total.
- Protected data: folds 8 and 9 remain unopened.
- Context: balanced head-tail truncation to 256 tokens.
- Schedule: one epoch, effective batch size 32, seed 42.
- A/B handling: 50% deterministic training swap and swap-averaged evaluation.
- Runtime gate: after eight training micro-batches, stop an arm whose projected
  training time exceeds three hours. This reserves time for both arms, their
  doubled A/B evaluation, setup, and export inside one 12-hour Kaggle session.

## Interpretation boundary

The pilot can answer two questions: whether both Gemma PEFT paths complete in
the available environment, and which arm is directionally stronger on the
same small balanced subset. It cannot establish that Gemma beats E2, and its
metrics must not enter the full-fold leaderboard.

Success means both arms complete and all aggregate metrics, runtimes, memory
measurements, ClearML provenance, and private reproducibility artifacts are
saved. No Kaggle submission is created by this experiment.
