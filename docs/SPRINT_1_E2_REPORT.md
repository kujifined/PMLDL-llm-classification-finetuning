# Sprint 1 — E2: Full fine-tuning, LoRA, and QLoRA

## Outcome

E2 is complete. On the frozen fold-7 selection protocol, QLoRA is the best
neural arm with **1.03429** validation log loss. It beats the strongest
classical local baseline (sparse TF-IDF, 1.04963) and the exploratory
classical blend (1.04761). The final fixed Kaggle ensemble improved the public
score from **1.02832** for QLoRA only to **1.02067**. A bounded follow-up then
selected a high-learning-rate rank-16 QLoRA adapter at **1.01627** on fold 7;
with the already fixed sparse blend it reached **1.01108** public Kaggle log
loss.

| Model or submission | Log loss | Evidence role |
|---|---:|---|
| Structural logistic regression | 1.06303 | local fold-7 baseline |
| Sparse TF-IDF | 1.04963 | local fold-7 baseline |
| Full fine-tuning, 3 epochs | 1.08737 | local fold-7 E2 arm |
| LoRA, 3 epochs | 1.04147 | local fold-7 E2 arm |
| QLoRA, 3 epochs | **1.03429** | local fold-7 E2 winner |
| QLoRA Kaggle submission | 1.02832 | Kaggle public score |
| Prior fixed 58% QLoRA + 42% sparse blend | 1.02067 | Kaggle public score |
| QLoRA HPO winner: r16, LR 2.8e-4, epoch 3 | **1.01627** | local fold-7 selection follow-up |
| Fixed 58% HPO QLoRA + 42% sparse blend | **1.01108** | Kaggle public score |

## Fair comparison

All E2 arms use `microsoft/deberta-v3-base` at revision `8ccc9b6f`, seed 42,
the same 40,235 training rows (folds 0-6), the same 5,746-row fold-7
validation set, maximum length 512, effective batch size 32, AdamW, balanced
head-and-tail truncation, random A/B swaps in training, and swap-averaged
inference. The factor under comparison is the fine-tuning method: all weights,
LoRA adapters, or LoRA adapters over a 4-bit quantised frozen backbone.

The LoRA screening used 500 steps per candidate. Rank and dropout had little
effect; learning rate `2e-4` was selected and then used by both
parameter-efficient arms. The full comparison used three epochs (3,774
optimizer steps per arm).

| Epochs | Full fine-tuning | LoRA | QLoRA |
|---:|---:|---:|---:|
| 1 | 1.08056 | 1.08061 | 1.08085 |
| 2 | 1.07514 | 1.07050 | 1.05878 |
| 3 | 1.08737 | 1.04147 | **1.03429** |

The methods separate only after more than one epoch. Full fine-tuning begins
to overfit after epoch two, while LoRA and QLoRA continue to improve. QLoRA
trains 1.18 million parameters (0.83% of the 184.4 million full model) and
matches or exceeds LoRA quality here.

## Bounded QLoRA follow-up and Kaggle inference

The E2 comparison itself remains the three-arm experiment above. Its separate
follow-up evaluated five pre-registered QLoRA candidates, all on folds 0-6 for
training and fold 7 for selection, with a five-epoch maximum and no access to
folds 8, 9 or Kaggle during training. `r=16`, dropout `0.05`, and learning
rate `2.8e-4` won at epoch three (1.01627); the epoch-four result was worse
(1.01883), so the epoch-three adapter was retained. The complete candidate
table and exact source evidence are in `docs/RESULTS.md` and
`docs/evidence/E20260909170000000000_hpo_summary.json`.

The [final private notebook](https://www.kaggle.com/code/karimkhabibrakhmanov/pmldl-e2-hpo-high-lr-qlora-sparse-blend)
runs with Internet off on a Kaggle T4. It verifies the HPO adapter payload
manifest and provenance, computes both the QLoRA and sparse probabilities for
the `test.csv` that Kaggle supplies, checks column order, IDs, finite values,
and probability sums, then writes `submission.csv`.

The original blend failed only because it stored sparse predictions for the
three local demonstration test IDs. The corrected notebook loads the frozen
sparse estimator and predicts whichever hidden test Kaggle supplies at runtime.
Its clean notebook commit completed successfully before it was submitted.

The 58/42 blend weight was selected before this HPO follow-up and stayed frozen.
The public score is external evaluation, not used to choose another trial,
epoch or blend weight.

## Limits and reproducibility

Fold 8 is still reserved for calibration and fold 9 for one final holdout, so
neither was opened during Sprint 1. The model shows raw A/B position sensitivity
before averaging; swap-averaging makes reported inference exactly symmetric.

The H100 run records local evidence under
`results/runs/E20260905212645934620__s42__38aea1ac__20260907T052625009035Z/`.
It passed the result, artifact, and run-schema checks. The environment used
`transformers` 4.56.2 because the internal mirror could not provide the locked
configuration; this and the safetensors re-serialisation are documented in
`infra/h100/README.md`.

ClearML could not be installed on the H100 mirror, so the original run metadata
says `tracking_status=unavailable`. A dedicated task in the shared project now
records this as a **historical import**, with the run identity, protocol,
three-arm results, Kaggle score, all 2,695 retained history points, and the
source JSON artifacts. `scripts/import_e2_to_clearml.py` is the reproducible
backfill. The imported task is not described as live tracking of the original
H100 execution.

The five HPO trials are documented in a second ClearML task as a **historical
import**, again preserving the distinction from live tracking. It contains the
five fold-7 results, the winner's epoch curve, the versioned JSON evidence and
the final Kaggle score. `scripts/import_hpo_to_clearml.py` recreates [the
task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/5539883a8cb84f1cb5e0f923de07b24c/general).
