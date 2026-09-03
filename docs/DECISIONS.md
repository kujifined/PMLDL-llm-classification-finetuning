# Decision log

## 2026-09-03: validation before leaderboard optimization

Decision: group all normalized duplicate prompts, reserve separate model
selection, calibration, and final holdout folds, and keep the final holdout
closed until the shortlist is frozen.

Reason: thousands of rows share prompts. A random row split would allow the
same user request on both sides and make local validation optimistic.

## 2026-09-03: exclude model identity

Decision: model_a and model_b may be used only for descriptive analysis, not
as training or inference features.

Reason: Kaggle test does not contain these columns. Depending on them creates
train-serving skew and a pipeline that cannot score the hidden test.

## 2026-09-03: enforce A/B symmetry

Decision: all trainable baselines receive swapped examples, and reported
inference averages original predictions with swapped-back predictions.

Reason: exchanging A and B should exchange the two winner probabilities while
leaving tie unchanged. This is an observable invariant and a direct defense
against position bias.

## 2026-09-03: freeze the realized split, not only its recipe

Decision: store one checked `id -> prompt_group -> fold` mapping and require
every experiment to load it. Record its hash together with data, config, and
code hashes in every metrics artifact.

Reason: a seed and splitter name do not guarantee identical assignments across
future library versions. A frozen mapping makes experiment comparisons stable.

## 2026-09-03: keep fold 8 calibration-only

Decision: freeze the exploratory sparse hyperparameter and blend weight after
fold 7. Fold 8 may fit temperature/tie-bias calibration only; it cannot be used
to select another architecture, alpha, or blend weight.

Reason: confirming a chosen weight by picking again on the calibration fold
would turn calibration into a second model-selection set and bias the final
holdout comparison.
