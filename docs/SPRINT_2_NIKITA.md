# Sprint 2 — Nikita: sparse TF-IDF strengthening

## Objective

Improve the existing E010 sparse preference model while preserving the project
protocol: prompt-grouped frozen folds, folds 0–6 for training, fold 7 for
selection, A/B swap augmentation, and swap-averaged inference. Fold 8 remains
reserved for calibration and fold 9 remains untouched until the final frozen
candidate is selected.

The model predicts the three competition classes in the order
`winner_model_a`, `winner_model_b`, `winner_tie`. Text features are built from
signed and absolute response-A/response-B TF-IDF differences, plus the existing
structural and prompt-response similarity features.

## Sequential sweep

The implementation is in `scripts/sweep_sparse_sprint2.py` and is configured by
`configs/experiments/E202609140001.json`. It changes one factor at a time:

1. analyzer and n-grams: word unigram/bigram control, word unigrams, word
   1–3-grams, character 3–5-grams, character 3–6-grams, and word+character;
2. vocabulary size: 50,000 and 100,000 around the best configuration;
3. `min_df`: 1 and 5 around the best configuration;
4. SGD `alpha`: 0.0001 and 0.001 around the best configuration.

Every candidate is evaluated on the same fold-7 validation set. The script
writes `sparse_sweep_comparison.csv`, logs the selected validation metrics, and
serializes the selected model, manifest, fold-7 predictions, and the three-row
test-schema prediction file under its `results/runs/<run_id>/` and
`artifacts/<run_id>/` directories.

## Execution

From the repository root:

```bash
PYTHONPATH=src python scripts/sweep_sparse_sprint2.py
```

The full sweep uses the configured 30 SGD epochs and is CPU-heavy. Before a
full run, the mechanics can be checked with:

```bash
PYTHONPATH=src python scripts/sweep_sparse_sprint2.py \
  --initial-candidate-limit 1 --skip-refinement --epochs 1
```

The smoke run is only a pipeline check and must not be presented as the final
model comparison. The local data directory is expected to be populated from
the competition archive; `scripts/audit_data.py` verifies its checksums before
the experiment starts.

## Current execution evidence

The data audit passed for the supplied competition archive. A one-candidate
smoke run completed and was validated with `scripts/validate_run.py`; it
produced the expected model manifest, serialized estimator, fold-7 prediction
file, and test-schema prediction file. Its metrics are intentionally not used
to select the final configuration because it used one epoch solely to test
the pipeline.

The full 30-epoch sweep should be run as a separate completed experiment after
the branch is committed, so its run metadata records a clean Git revision.
