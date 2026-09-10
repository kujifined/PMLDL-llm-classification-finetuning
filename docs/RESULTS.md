# Verified results

This page contains only results backed by versioned metric files under
`results/`. It is a working record, not the final report.

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

- results/baselines/bias_baseline/evaluation.json
- artifacts/bias_baseline/validation_predictions.csv
- data/splits/folds.csv
- data/splits/metadata.json
- artifacts/bias_baseline/evaluation_model.joblib
- artifacts/bias_baseline/full_model.joblib
- artifacts/bias_baseline/test_schema_predictions.csv
- results/baselines/sparse_baseline/evaluation.json
- artifacts/sparse_baseline/validation_predictions.csv
- artifacts/sparse_baseline/evaluation_model.joblib
- artifacts/sparse_baseline/test_schema_predictions.csv
- results/baselines/baseline_blend/evaluation.json
- artifacts/baseline_blend/validation_predictions.csv

Run `make check-results` to verify all 15 displayed metric cells against the
three aggregate metric results. Run `make baseline`, `make sparse-baseline`,
`make blend-baselines`, and `make verify-artifacts` to reconstruct and verify
the ignored evidence files.

## E060: exact swap-equivariant pair encoder (informal result)

First reported neural result, run on Kaggle GPU with
`scripts/train_symmetric_encoder.py` and `configs/symmetric_encoder.json`
(`microsoft/deberta-v3-small`, max length 256, batch size 16, 1 epoch). Same
validation protocol as E000-E011 above: folds 0-6 for training, fold 7 for
validation, 40,235 training rows, 5,746 validation rows.

Architecture: a shared encoder embeds `prompt + response_A` and
`prompt + response_B` independently into `h_a` and `h_b`. The preference logit
is a bias-free linear map of `h_a - h_b` (odd by construction, so swapping A
and B flips its sign exactly), and the tie logit is a small MLP applied to
`|h_a - h_b|` (invariant to the swap by construction). Unlike E000-E011, swap
symmetry here is architectural, not obtained by averaging two inference
passes.

| Experiment | Validation log loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| E060 Swap-equivariant pair encoder (deberta-v3-small, 1 epoch) | 1.050053 | 0.455621 | 0.456849 |

For context against the table above: E060 clears E002 (1.063033) and is a
statistical tie with E010 (1.049629) and E011 (1.047605) -- the gap is in the
third decimal, on par with E011's own bootstrap interval width, so this is
not yet a demonstrated improvement over the sparse baseline. One epoch is a
lower bound, not a ceiling: the model ladder calls for E030 (a tuned
cross-encoder), E040 (balanced truncation + A/B swap training), and E050
(swap-averaged inference + calibration) before E060, and none of those are
reported yet, so this stretch architecture has not had the same tuning budget
as E010/E011.

Diagnostics from `results/baselines/symmetric_encoder/evaluation.json`:

- raw (un-averaged) A/B symmetry error: 4.25e-9 mean L1 -- nine orders of
  magnitude tighter than E002/E010's post-hoc-averaged 3.96e-5, and obtained
  without averaging two passes, which is the architectural claim this
  experiment tests;
- ECE-15: 0.0748 (uncalibrated; folds 8/9 are reserved for calibration);
- predicted class rates: [0.262 A, 0.257 B, 0.482 tie] against true rates
  [0.349, 0.342, 0.309] -- the model over-predicts tie;
- confusion matrix (rows = true A/B/tie, columns = predicted A/B/tie):
  `[[801, 292, 913], [310, 808, 847], [392, 374, 1009]]`.

Runtime was 6,048 seconds (~100.8 minutes) on a Kaggle GPU.

**Reproducibility gap -- not yet leaderboard-eligible.** This run predates the
project's `ExperimentRun` / `results/runs/` pipeline (see
`docs/NOTEBOOK_WORKFLOW.md`) and was recorded from Git commit
`349c083d8c5958d4a6f777f0ce5931d4080c7a73` with a **dirty** working tree.
`configs/project.json`'s `run_policy.require_clean_git` means
`ExperimentRun` itself refuses to start a non-smoke run on a dirty tree, so
this result is committed as reviewable legacy-style evidence under
`results/baselines/symmetric_encoder/` (mirroring the E000-E011 baselines)
rather than as a `results/runs/E060__...` entry in `results/leaderboard.csv`.
`configs/experiments/E060.json` records the experiment definition. Before
E060 can be compared to future runs on the leaderboard, re-run it from a
clean commit -- ideally through `make new-experiment` so it gets an
`ExperimentRun`-tracked run directory, ClearML task, and committed
`config.json`/`run.json`/`metrics.json` triple.

Evidence:

- results/baselines/symmetric_encoder/evaluation.json
- results/baselines/symmetric_encoder/validation_predictions.csv (committed,
  not gitignored, since no local checkpoint exists to regenerate it)
- configs/experiments/E060.json
- configs/symmetric_encoder.json
- data/splits/folds.csv
- data/splits/metadata.json
