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

## E2: full fine-tuning versus LoRA versus QLoRA

One H100 80 GB, `microsoft/deberta-v3-base` at revision `8ccc9b6f`, seed 42,
folds 0-6 for training and fold 7 for comparison, 40,235 training rows and
5,746 validation rows, maximum length 512, three epochs, 3,774 optimizer steps
per arm, effective batch size 32, AdamW in fp32 with the forward pass in bf16,
balanced head-and-tail truncation, random A/B swap during training and
swap-averaged inference. Folds 8 and 9 remain unopened.

### LoRA screening, 500 steps each on fold 7

| Candidate | r | dropout | learning rate | Log loss |
|---|---:|---:|---:|---:|
| reference | 16 | 0.05 | 1e-4 | 1.089734 |
| rank 8 | 8 | 0.05 | 1e-4 | 1.089819 |
| dropout 0 | 16 | 0.00 | 1e-4 | 1.089493 |
| learning rate 2e-4 | 16 | 0.05 | 2e-4 | **1.087956** |

Rank and dropout barely move the metric; the learning rate does. Both
parameter-efficient arms use the winning candidate.

### Arms

| Arm | Log loss | Accuracy | Macro-F1 | ECE-15 | Brier | Swap error | Trainable | Share | Peak GPU | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full fine-tuning | 1.08737 | 0.4346 | 0.4330 | 0.0465 | 0.6577 | 0.15079 | 184,424,451 | 100% | 6531 MB | 941 s |
| LoRA | 1.04147 | 0.4575 | 0.4558 | **0.0108** | 0.6272 | 0.19231 | 1,182,723 | 0.64% | 5942 MB | 928 s |
| QLoRA | **1.03429** | **0.4617** | **0.4590** | 0.0116 | **0.6221** | 0.16203 | 1,182,723 | 0.83% | 5680 MB | 945 s |

QLoRA is the best arm at 1.03429, ahead of the sparse baseline at 1.0496 and
the blend at 1.0476. This is the first neural result in the project that beats
the classical baselines.

### Training duration decides this comparison

The same three arms were run at one, two and three epochs, changing nothing
else:

| Epochs | Full fine-tuning | LoRA | QLoRA |
|---:|---:|---:|---:|
| 1 | 1.08056 | 1.08061 | 1.08085 |
| 2 | 1.07514 | 1.07050 | 1.05878 |
| 3 | 1.08737 | **1.04147** | **1.03429** |

At one epoch the three methods are indistinguishable and all three lose to the
classical baselines. The ordering only appears with a longer budget, and it
reverses for full fine-tuning: it improves from one to two epochs and then
degrades at three, while both parameter-efficient arms keep improving. Updating
184 million parameters overfits this 40,235-row training set sooner than
updating 1.18 million does.

So the headline of E2 is not that LoRA is a cheaper approximation of full
fine-tuning. On this task it is the better model, and quantising its frozen
backbone to 4 bits costs nothing in quality.

An earlier version of these runs trained the backbone in bf16 and updated it
directly with AdamW. At a learning rate of 2e-5 an update is roughly 0.1% of a
weight while bf16 resolves about 0.4%, so most updates rounded away and every
arm stalled near 1.09. Master weights are now fp32 with the forward pass under
autocast.

### What the run does not show

Peak memory is 5680-6531 MB across the arms, a spread of 15%. Activations
dominate at this model size and batch shape, so freezing and quantising the
backbone barely shrinks the footprint. Training time is within 2%. The
practical argument for QLoRA is about larger models than this one.

Raw A/B asymmetry grows with training: 0.15 to 0.19 mean L1 at three epochs
against 0.02 to 0.05 at one. The longer these models train, the more they
respond to the position of an answer rather than its content. The reported
probabilities are exactly symmetric because validation averages the original
and swapped orders, so the metric hides it, but the underlying model is
order-sensitive and that is a fragility on the hidden test set.

Evidence: `results/runs/E20260905212645934620__s42__38aea1ac__20260907T052625009035Z/`
and the matching ignored `artifacts/` directory, which hold `study.json`,
per-arm training curves, validation predictions, and the three checkpoints. The
one- and two-epoch runs are under `810967c4` and `02396816`.

The commit hashes recorded inside those run directories were produced on a
branch that also carried an unrelated follow-up study. The four code commits
were cherry-picked here, so each carries a `cherry picked from commit` trailer
that maps it to the hash its runs recorded.

### Environment deviations

The run did not reproduce the pinned locks and its artifacts record what
actually executed. `transformers` was held at 4.56.2 instead of the locked
4.57.6, `requirements-baseline.lock` was not applied because the internal
mirror lacks those versions, ClearML is not published on that mirror so
tracking status is `unavailable`, and the pinned checkpoint was re-serialised
from `pytorch_model.bin` to safetensors without changing revision or weights.
The sprint plan sketched one to two epochs and this run uses three.
`infra/h100/README.md` explains each one.

### Bounded QLoRA follow-up: five pre-registered candidates

After the E2 arm comparison was complete, a separate, bounded follow-up ran
five QLoRA candidates on the same frozen fold-7 selection protocol. It changed
only LoRA rank, dropout, learning rate and the stopping epoch. The candidate
list, five-epoch ceiling and stopping rule were committed before the runs;
Kaggle, fold 8 and fold 9 were not read during this search.

| Candidate | r | dropout | learning rate | Best epoch | Fold-7 log loss |
|---|---:|---:|---:|---:|---:|
| reference_r16 | 16 | 0.05 | 2.0e-4 | 3 | 1.02507 |
| low_lr_r16 | 16 | 0.05 | 1.2e-4 | 5 | 1.04791 |
| high_lr_r16 | 16 | 0.05 | 2.8e-4 | 3 | **1.01627** |
| rank32 | 32 | 0.05 | 2.0e-4 | 2 | 1.03805 |
| rank32_dropout10 | 32 | 0.10 | 2.0e-4 | 1 | 1.08046 |

The selected `high_lr_r16` candidate improves through epoch three
(1.08153, 1.03445, **1.01627**) and rises to 1.01883 at epoch four. Its
epoch-three accuracy is 0.48556, macro-F1 is 0.48177, ECE-15 is 0.01554, and
raw A/B swap L1 is 0.15953. This is a selection-fold result, not an untouched
holdout result. The versioned summary contains the exact values, protocol,
Nirvana workflow and Kaggle submission reference:
`docs/evidence/E20260909170000000000_hpo_summary.json`.

### Kaggle inference and public score

The final Internet-Off Kaggle inference notebook freezes the selected HPO
three-epoch QLoRA adapter and combines its swap-averaged probabilities with
the frozen sparse baseline: 58% QLoRA and 42% sparse. This weight had already
been fixed on fold 7; it was not re-tuned for the HPO result or on Kaggle.

The first blend notebook incorrectly carried predictions for the three local
demonstration test IDs. Kaggle substitutes the hidden `test.csv`, so that
version failed during the private re-run. The corrected version loads the
frozen sparse model and computes sparse predictions from the supplied test at
runtime. It was saved as a fresh Kaggle version, successfully re-run in a
clean Kaggle T4 environment, and then submitted.

| Kaggle submission | Public log loss | Status |
|---|---:|---|
| QLoRA only, three epochs | 1.02832 | completed |
| Prior fixed 58% QLoRA + 42% sparse blend | 1.02067 | completed |
| Fixed 58% HPO QLoRA + 42% sparse blend | **1.01108** | completed |

The HPO blend improves the prior fixed-blend public score by 0.00959 log loss.
This is external evaluation evidence, not an additional local selection signal:
folds 8 and 9 remain unopened. The notebook is available at
<https://www.kaggle.com/code/karimkhabibrakhmanov/pmldl-e2-hpo-high-lr-qlora-sparse-blend>.

## Sprint 2 E2: fixed three-seed follow-up

The Sprint 2 continuation did not start a new broad search. It froze the HPO
winner (`r=16`, alpha 32, dropout 0.05, learning rate `2.8e-4`) and applied the
assigned epoch and seed-stability checks on fold 7. The earlier HPO trajectory
already supplied the exact epoch-three versus epoch-four comparison under the
same protocol: log loss rose from 1.01627 to 1.01883, so the rule "run epoch
five only if epoch four improves" stopped training there.

Three clean epoch-three jobs then changed only the seed:

| Candidate | Log loss | Accuracy | Macro-F1 | ECE-15 | Brier | Swap error |
|---|---:|---:|---:|---:|---:|---:|
| seed 42 | **1.01505** | **0.48747** | **0.48420** | 0.01583 | **0.60864** | 0.16376 |
| three-seed probability mean | 1.01889 | 0.48016 | 0.48001 | **0.01547** | 0.61150 | **0.13605** |
| seed 17 | 1.02800 | 0.46711 | 0.46777 | 0.03340 | 0.61816 | 0.19701 |
| seed 73 | 1.02807 | 0.47355 | 0.47328 | 0.01775 | 0.61781 | 0.16072 |

The ensemble is 0.00384 worse than the best single seed, although its raw A/B
asymmetry is lower. Seed 42 is therefore the handoff candidate. No ensemble
weight was tuned, and fold 8, fold 9 and Kaggle were not read.

The three jobs ran from clean commit `e76c1fcb` in one
[Nirvana process](https://nirvana.yandex-team.ru/process/d5e9ed94-7877-4570-814d-285ee16ca215).
Their original ClearML status remains `unavailable`; the complete retained
evidence was uploaded separately to [the explicitly historical task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/b158f2d9a9394b68bd7188459e0b375e/general).
The detailed handoff record is `docs/SPRINT_2_E2_MULTI_SEED_REPORT.md`, while
the machine-readable comparison is `results/e2_multiseed/summary.json`.
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
