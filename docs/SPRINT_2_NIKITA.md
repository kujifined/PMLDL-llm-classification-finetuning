# Sprint 2 — Nikita: sparse TF-IDF strengthening

## Objective

Improve the verified B2 word+character TF-IDF experiment
`E20260909125010457517`. Its clean full run is the mandatory control:
40,235 training rows, 5,746 fold-7 validation rows, seed 42, and validation log
loss 1.040515. The Sprint 2 run preserves the prompt-grouped frozen split, A/B
swap augmentation, and swap-averaged inference. Fold 8 remains reserved for
calibration and fold 9 remains untouched until the final candidate is frozen.

The model predicts the three competition classes in the order
`winner_model_a`, `winner_model_b`, `winner_tie`. Text features are built from
signed and absolute response-A/response-B TF-IDF differences, plus the existing
structural and prompt-response similarity features.

## Sequential sweep

The implementation is in `scripts/sweep_sparse_sprint2.py` and is configured by
`configs/experiments/E202609140001.json`. It changes one factor at a time:

1. word n-grams: `(1, 1)` and `(1, 3)` against the B2 `(1, 2)` control;
2. `char_wb` n-grams: `(2, 5)` and `(3, 6)` against `(3, 5)`;
3. word vocabulary size: 50,000 and 100,000 against 75,000;
4. character vocabulary size: 25,000 and 75,000 against 50,000;
5. word `min_df`: 1 and 5 against 3;
6. character `min_df`: 3 and 8 against 5;
7. SGD `alpha`: 0.001 and 0.0001 against 0.0003.

Each stage starts from the best configuration selected by the preceding stage,
and each candidate changes exactly one field. The implementation imports the
same `WordCharacterFeatureEncoder` used by B2 rather than duplicating its
feature logic.

Every candidate is evaluated on the same fold-7 validation set. The script
writes `sparse_sweep_comparison.csv` after every candidate, saves
`best_tfidf_config.json`, logs the selected validation metrics and deltas from
B2, and serializes the selected model, manifest, fold-7 predictions, and the
three-row test-schema prediction file under its `results/runs/<run_id>/` and
`artifacts/<run_id>/` directories.

## Execution

From the repository root:

```bash
PYTHONPATH=src python scripts/sweep_sparse_sprint2.py
```

The full sweep uses the B2 classifier protocol with 30 SGD iterations and is
CPU/RAM-heavy. The local data directory is expected to be populated from the
competition archive; `scripts/audit_data.py` verifies its checksums before the
experiment starts.

## Current execution status

The exact B2 source, tests, full run record, and parent metrics are included in
the Sprint 2 branch. The earlier one-epoch smoke result predates this B2-based
implementation and was removed from the final tree; it remains recoverable
from Git history but must not be used in the final comparison or report. The
next canonical result is a clean full run from the committed branch.

For a Kaggle Notebook, attach the competition data and run the repository from
`/kaggle/working`. Kaggle exposes the competition CSV files through a read-only
input directory, so use the prepared
`output/kaggle/run_sparse_sprint2.ipynb` notebook or the external-data command:

```bash
python scripts/sweep_sparse_sprint2.py \
  --data-dir /kaggle/input/llm-classification-finetuning \
  --external-data
```

This mode still checks the SHA-256 hashes of `train.csv`, `test.csv`, and
`sample_submission.csv`; it only skips requiring the original ZIP inside the
Git clone.

The prepared notebook reads ClearML credentials from private Kaggle Secrets.
The full run is logged to the shared `PMLDL LLM Classification Finetuning`
project with its configuration, candidate metrics, selected-model metrics, and
handoff artifacts. Credential values are neither printed nor committed.

After the run, validate the generated run directory and use its comparison CSV,
best-config JSON, metrics, model manifest, and fold-7 predictions for the PR and
handoff. Do not use the Kaggle public leaderboard to select the sparse
configuration.
