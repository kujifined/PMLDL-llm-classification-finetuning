# LLM Preference Classification

This is the anonymized final release for the PMLDL course project. The task is
to predict whether a person prefers the response of Model A, the response of
Model B, or neither response (a tie) for each dialogue in the Kaggle
competition [LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning).

The primary metric is multiclass log loss. Lower is better.

## Frozen evaluation protocol

- Folds 0--6: training.
- Fold 7: candidate selection and controlled ablations.
- Fold 8: blend-weight search and temperature calibration.
- Fold 9: one-time final holdout evaluation after the final configuration is
  frozen.

The frozen ensemble is defined in `configs/final_model.json`:

- DeBERTa-v3-base QLoRA, seed 42, weight 0.5;
- improved word + character sparse TF-IDF, weight 0.5;
- multiclass temperature scaling with `T = 0.8173109972`.

Its fold-8 calibration result is `Log loss = 0.996973`. This is not a final
holdout result: fold 9 is intentionally not evaluated in the committed state.

## Repository contents

```text
configs/     frozen folds, selected model configurations, and ensemble settings
data/        data acquisition instructions, checksums, and fixed fold mapping
src/         reusable data, split, feature, metric, and submission code
scripts/     baseline, QLoRA, sparse, calibration, and finalization entry points
results/     anonymized comparison tables and final ensemble search evidence
artifacts/   documented external checkpoints and prediction inputs; not in Git
docs/        exact reproduction, attribution, and packaging instructions
tests/       data-independent checks for the release code
```

The raw competition data, model checkpoints, probability files, and the final
submission are intentionally excluded from Git. Their required filenames,
checksums, and retrieval locations are listed in
[`artifacts/FINAL_ARTIFACTS.md`](artifacts/FINAL_ARTIFACTS.md).

## Quick start

Use Python 3.12. Download the competition data only after accepting the Kaggle
rules; see `data/README.md` for the expected local layout.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.lock
python -m pip install -r requirements-transformer.lock
python -m pip install -e . --no-deps

make test
make audit
```

The complete, ordered procedure for regenerating the final metrics and
`submission.csv` is in `docs/REPRODUCE.md`.
