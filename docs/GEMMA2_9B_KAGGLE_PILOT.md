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

## Executed pilot: 14 September 2026

The pilot itself **completed both arms** on Kaggle T4 x2. The completed run is
`E20260914071959000000__s42__95c2d20c__20260914T080556416319Z`, from clean
Git commit `95c2d20cad2024db18d6005ee9127d9fcd9ecedd`. The experiment ran
from 08:05:56 to 12:19:29 UTC (4 h 13 m). ClearML tracking completed online
and closed normally; its workspace-specific task identifier is redacted from
the public repository.

| Arm | Fold-7 log loss ↓ | Accuracy | Macro-F1 | A/B swap error L1 ↓ | Arm runtime | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 1.091112 | 0.394444 | 0.385729 | 0.372937 | 1 h 55 m 33 s | 9,083 MB |
| QLoRA | 1.089694 | 0.372222 | 0.370047 | 0.432368 | 2 h 16 m 33 s | 9,363 MB |

Both arms reached epoch 1 and 29 optimizer steps. QLoRA was selected by the
preregistered primary metric (log loss), but its advantage is only **0.001417**
on 180 balanced validation rows. LoRA has higher accuracy and macro-F1 and
lower A/B swap error here. This single small-subset run does not establish a
robust winner or justify replacing the full-fold E2/E4 results. In particular,
swap error is high for both arms despite swap-averaged inference; investigate
the positional sensitivity before using either model downstream.

The Kaggle notebook's overall version is marked **failed** because an old,
unrelated diagnostic cell after the launcher raised `IndexError` while
indexing an empty preflight list for the parent experiment. The launcher and
the actual experiment cell completed first, wrote the run, and saved
`/kaggle/working/gemma2-pilot-output.zip`. This is a notebook-packaging error,
not an arm-training failure. Remove the stale diagnostics before a future
`Save & Run All`; do not rerun this four-hour experiment merely to make the
version badge green.

Aggregate records are in `results/runs/<run_id>/{config,run,metrics,study}.json`.
The run, metrics, and study values were transcribed from Kaggle's rendered
JSON previews and validated locally; the config is the same immutable source
configuration loaded by the clean run. Kaggle retains the original JSONs and
private 414.83 MB output archive. The archive was not imported into this
checkout, so byte-for-byte equality and
the model/prediction artifact integrity have **not** been independently checked.
Row-level predictions and model checkpoints must remain out of Git. No Kaggle
competition submission was created.
