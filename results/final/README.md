# Final ensemble protocol

The candidate boundary is frozen on fold 7 before any fold-8 metric is used:
a candidate is competitive when its absolute log-loss delta from the DeBERTa
QLoRA seed-42 anchor is at most 0.05. DeBERTa and the improved sparse model are
available for the final search. The original one-epoch E060 symmetric model
passed the metric boundary but is excluded because its selected checkpoint was
not retained; the epoch-2 model is a different candidate and failed the fold-7
boundary. Gemma-2 9B QLoRA is excluded before calibration.

Gemma was a resource-bounded 180-row balanced pilot rather than a full fold-7
run. `pilot_subset_comparison.csv` therefore compares Gemma, DeBERTa, and E060
only on the exact shared IDs. Pilot rows are never blended with full-fold
predictions.

Place owner-provided prediction files under `artifacts/final_inputs/` using the
paths in `configs/final_model_search.json`. Files must contain aligned `id`,
`target`, `winner_model_a`, `winner_model_b`, and `winner_tie` columns.

Run the auditable stages separately:

```bash
python scripts/calibrate_and_blend.py \
  --stage select \
  --config configs/final_model_search.json \
  --output-dir results/final

python scripts/calibrate_and_blend.py \
  --stage blend \
  --config configs/final_model_search.json \
  --output-dir results/final
```

The blend stage checks three positive pair weights per pair and four positive
triple weights: equal weights plus the three permutations of 0.25/0.25/0.50.
It fits one bounded scalar temperature for each blend on fold 8, writes the
complete grid and comparison table, freezes `configs/final_model.json`, and
writes fold-8 verification probabilities under ignored
`artifacts/final_output/`. Fold 9 must remain unopened until that file is
frozen.

After owners provide inference probabilities for the same checkpoints, create
the hand-off probabilities without changing weights or temperature:

```bash
python scripts/calibrate_and_blend.py \
  --stage apply \
  --config configs/final_model_search.json \
  --output-dir artifacts/final_output
```
