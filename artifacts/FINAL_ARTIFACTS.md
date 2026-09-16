# External artefacts required for the final pipeline

Large artefacts are intentionally not committed. Restore them from the listed
ClearML tasks or from the model owners' hand-off storage, then verify SHA-256
before inference.

| Artefact | Retrieval reference | SHA-256 | Required for |
|---|---|---|
| DeBERTa QLoRA adapter, seed 42 | `clearml:b158f2d9a9394b68bd7188459e0b375e/E20260914010000000000__s42__e76c1fcb__20260913T215932089199Z_qlora_adapter.zip` | `d2bebee99d24417aa26435d1126f20a30ffeb41466db19ea20dca64177ead4a6` | DeBERTa inference |
| Sparse TF-IDF estimator | `clearml:89ac2f0b9c764ff6a57ec4d93be6a356/sparse_estimator.joblib` | `5be06b0efdd19191f69e96838484fe80503d9a7cc8dd4494c4d738e610cf8559` | Sparse inference |
| DeBERTa fold-7 and fold-8 probabilities | `artifacts/final_inputs/deberta_qlora_seed42/` | See `configs/final_model.json` | Calibration audit |
| Sparse fold-7 and fold-8 probabilities | `artifacts/final_inputs/sparse_tfidf_sprint2/` | See `configs/final_model.json` | Calibration audit |
| DeBERTa fold-9 and Kaggle-test probabilities | `artifacts/final_inputs/deberta_qlora_seed42/` | Pending final inference | Final evaluation and submission |
| Sparse fold-9 and Kaggle-test probabilities | `artifacts/final_inputs/sparse_tfidf_sprint2/` | Pending final inference | Final evaluation and submission |

The two pending rows are the only model outputs still required before the
final holdout can be opened and `submission.csv` can be created. The expected
CSV schemas and exact commands are in `docs/REPRODUCE.md`.
