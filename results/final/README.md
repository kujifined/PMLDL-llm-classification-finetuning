# Final ensemble protocol

The candidate boundary is frozen on fold 7 before any fold-8 metric is used.
The final search contains only the two eligible full-fold candidates: DeBERTa
QLoRA seed 42 and the improved sparse TF-IDF model. Their frozen weights and
the calibration temperature are recorded in `configs/final_model.json`.

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

The blend stage checks the declared pair-weight grid, fits one bounded scalar
temperature for each candidate blend on fold 8, writes the complete grid and
comparison table, freezes `configs/final_model.json`, and writes fold-8
verification probabilities under ignored
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
