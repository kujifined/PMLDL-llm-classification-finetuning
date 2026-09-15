# Final ensemble protocol

The candidate boundary is frozen on fold 7 before any fold-8 prediction is
loaded: a candidate is competitive when its absolute log-loss delta from the
DeBERTa QLoRA seed-42 anchor is at most 0.05. The threshold admits DeBERTa,
the improved sparse model, and the original one-epoch E060 symmetric model.
Gemma-2 9B QLoRA is excluded before calibration.

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
complete grid and comparison table, and freezes `configs/final_model.json`.
Fold 9 must remain unopened until that file is frozen.

After owners provide inference probabilities for the same checkpoints, create
the hand-off probabilities without changing weights or temperature:

```bash
python scripts/calibrate_and_blend.py \
  --stage apply \
  --config configs/final_model_search.json \
  --output-dir artifacts/final_output
```
