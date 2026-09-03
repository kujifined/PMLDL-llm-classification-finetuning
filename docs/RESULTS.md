# Verified results

This page contains only results backed by files under artifacts/. It is a
working record, not the final report.

## E000-E011: initial baselines

Validation protocol:

- ten prompt-grouped folds with seed 20260903;
- folds 0-6 used for training;
- fold 7 used for this comparison;
- folds 8 and 9 remain reserved for calibration and final evaluation;
- 40,235 training rows and 5,746 validation rows;
- model identity excluded;
- A/B swap augmentation and swap-averaged inference enabled.

| Experiment | Validation log loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| E000 Uniform | 1.098612 | 0.349112 | 0.172515 |
| E001 Train prior | 1.097216 | 0.349112 | 0.172515 |
| E002 Structural Logistic Regression | 1.063033 | 0.446920 | 0.440137 |
| E010 Sparse TF-IDF, alpha 0.0003 | 1.049629 | 0.467282 | 0.456832 |
| E011 15% structural + 85% sparse | 1.047605 | 0.464671 | 0.454038 |

E002 uses 56 structural features describing length, turns, empty/null turns,
formatting, URLs, digits, and A/B differences. Its improvement over the prior
confirms that surface-form preference biases are measurable, but this model
does not evaluate semantic answer quality.

Raw A/B symmetry error before inference averaging is 0.0000396 mean L1. The
reported prediction is exactly symmetric by construction after averaging.

E010 uses a 75,000-term word unigram/bigram vocabulary, signed and absolute
TF-IDF differences between responses, prompt-response cosine similarities,
and the structural features. The regularization search tested alpha values
0.001, 0.0003, and 0.0001; 0.0003 minimized selection-fold log loss.

E011 selects its blend weight on the same selection fold, so it is a frozen
candidate, not independent evidence. A 2,000-resample prompt-grouped paired
bootstrap estimates the log-loss delta versus E010 as -0.00202 with a 95%
interval from -0.00449 to -0.000003. Because the 101 blend weights were searched
on this fold, that interval does not correct for selection bias. Fold 8 may be
used only to fit probability calibration after base models are refit on folds
0-7; it must not be used to retune the blend weight.

Evidence (aggregate metrics are versioned; checkpoints and row-level
predictions are generated locally and intentionally ignored by Git):

- artifacts/bias_baseline/metrics.json
- artifacts/bias_baseline/validation_predictions.csv
- data/splits/folds.csv
- data/splits/metadata.json
- artifacts/bias_baseline/evaluation_model.joblib
- artifacts/bias_baseline/full_model.joblib
- artifacts/bias_baseline/test_schema_predictions.csv
- artifacts/sparse_baseline/metrics.json
- artifacts/sparse_baseline/validation_predictions.csv
- artifacts/sparse_baseline/evaluation_model.joblib
- artifacts/sparse_baseline/test_schema_predictions.csv
- artifacts/baseline_blend/metrics.json
- artifacts/baseline_blend/validation_predictions.csv

Run `make check-results` to verify all 15 displayed metric cells against the
three aggregate metric artifacts. Run `make baseline`, `make sparse-baseline`,
`make blend-baselines`, and `make verify-artifacts` to reconstruct and verify
the ignored evidence files.
