# Committed results

This release keeps only compact, reviewable evidence for the final solution:

- `baselines/`: the two reference baselines;
- `qlora_hyperparameter_search.csv`: the pre-registered QLoRA search;
- `e2_multiseed/`: DeBERTa seed comparison and calibration handoff metadata;
- `final/`: selection, calibration, ensemble-search, and final-holdout files;
- `model_comparison.csv`: the one-table summary used in the technical report.

Raw data, checkpoints, per-row prediction files, and the final submission are
excluded from Git. Their required locations and integrity hashes are listed in
`artifacts/FINAL_ARTIFACTS.md` and `docs/REPRODUCE.md`.
