# Reproducing the final release

This document reproduces the frozen final ensemble without using fold 9 to
select models, blend weights, or calibration parameters.

## 1. Prepare the environment and data

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.lock
python -m pip install -r requirements-transformer.lock
python -m pip install -e . --no-deps
make audit
```

Download the competition files into `data/llm-classification-finetuning/` as
documented in `data/README.md`. `make audit` verifies their checksums and the
fixed fold assignment in `data/splits/folds.csv`.

## 2. Reproduce the QLoRA selection search

The five fixed QLoRA candidates are listed in
`configs/experiments/qlora_hyperparameter_search.json`; their recorded fold-7
results are in `results/qlora_hyperparameter_search.csv`. The historical
runner requires an NVIDIA H100 GPU; it fails explicitly on different hardware
so that the comparison is not silently changed. To run one candidate:

```bash
python scripts/run_qlora_hpo_trial.py \
  --config configs/experiments/qlora_hyperparameter_search.json \
  --trial-id high_lr_r16
```

Run the same command for each listed `trial-id`, then compare only their
fold-7 log loss. Fold 8 and fold 9 must remain unused during this step.

## 3. Restore external artefacts

Retrieve exactly the two frozen checkpoints and the prediction files listed in
`artifacts/FINAL_ARTIFACTS.md`. Never replace a checkpoint, retrain a model, or
change `configs/final_model.json` after fold 8.

For a complete independent rerun of candidate selection and calibration, place
the following raw prediction files under `artifacts/final_inputs/` for both
models:

```text
fold7_predictions.csv   # id, target, winner_model_a, winner_model_b, winner_tie
fold8_predictions.csv   # id, target, winner_model_a, winner_model_b, winner_tie
```

Then execute:

```bash
make calibrate
```

This must reproduce `results/final/ensemble_grid.csv`,
`results/final/final_comparison.csv`, and the frozen blend in
`configs/final_model.json`.

## 4. Final holdout and Kaggle submission

Only after the model configuration is frozen, place the raw, uncalibrated
probabilities from the same two checkpoints in the locations below:

```text
artifacts/final_inputs/deberta_qlora_seed42/fold9_predictions.csv
artifacts/final_inputs/deberta_qlora_seed42/inference_predictions.csv
artifacts/final_inputs/sparse_tfidf_sprint2/fold9_predictions.csv
artifacts/final_inputs/sparse_tfidf_sprint2/inference_predictions.csv
```

`fold9_predictions.csv` must contain `id`, `target`, and the three class
probabilities. `inference_predictions.csv` must contain `id` and the three
class probabilities. In both files the classes are ordered as Model A, Model B,
tie; values are finite, non-negative, and sum to one in every row.

Run:

```bash
make finalize FINALIZE_ARGS="--fold9-predictions <deberta_fold9.csv> <sparse_fold9.csv> --test-predictions <deberta_test.csv> <sparse_test.csv>"
```

The command uses only the stored weights and temperature. It writes:

```text
results/final/fold9_final_probabilities.csv
results/final/final_holdout_metrics.json
results/final/submission.csv
```

Validate the `submission.csv` schema before uploading it manually to Kaggle.
Record the public score separately from the frozen-fold metrics.

`submission.csv` is a generated competition artefact and is deliberately not
included in the course `project.zip`.

## 5. Package the project

Compile the anonymized `project.pdf`, run the anonymity scan in
`docs/SUBMISSION_CHECKLIST.md`, and build the archive from tracked files only:

```bash
git archive --format=zip --output project.zip HEAD
```

Do not add raw Kaggle data, API tokens, checkpoints, ClearML credentials,
personal paths, names, usernames, or the `.git` directory to the archive.
