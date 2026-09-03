# Experiment plan

## Objective

Minimize multiclass log loss for the classes A, B, and tie while preserving
honest model selection, swap symmetry, reproducibility, and offline Kaggle
inference below the nine-hour limit.

## Locked data protocol

The first project decision is a fixed 10-fold StratifiedGroupKFold split.
The group key is SHA-256 of the normalized JSON prompt. Unicode NFKC,
whitespace normalization, case folding, and turn boundaries are part of the
key. Model text itself remains unnormalized.

The realized assignment is stored in `data/splits/folds.csv` with its checksum
and generator environment in `data/splits/metadata.json`. Training scripts fail
closed if ids, prompt hashes, fold coverage, dataset checksums, or role config
do not match.

Split roles:

| Folds | Role | Allowed use |
|---|---|---|
| 0-6 | Training | Fit model parameters |
| 7 | Selection | Early stopping and controlled ablations |
| 8 | Calibration | Temperature and optional tie-bias calibration |
| 9 | Final holdout | One evaluation after the shortlist is frozen |

The frozen split has also been checked for canonical unordered-battle leakage;
no identical prompt plus response pair crosses partitions.

## Model ladder

| ID | Model | Purpose |
|---|---|---|
| E000 | Uniform probability | Metric and submission sanity check |
| E001 | Training class prior | Class-imbalance baseline |
| E002 | Structural Logistic Regression | Quantify length and formatting bias |
| E010 | Word unigram/bigram TF-IDF | Strong sparse-text baseline |
| E020 | Public KerasNLP starter reproduction | Existing neural solution 1 |
| E021 | Public DeBERTa notebook reproduction | Existing neural solution 2 |
| E030 | Fine-tuned DeBERTa cross-encoder | Main proposed model |
| E040 | Balanced truncation plus A/B swap training | Controlled improvement |
| E050 | Swap-averaged inference plus temperature scaling | Calibration improvement |
| E060 | Exact swap-equivariant pair encoder | Stretch architecture |
| E070 | Two-seed probability ensemble | Final option if gain exceeds noise |

E020 must retain the original source and license. E030 onward must be
implemented as the team's own controlled work, with every borrowed component
attributed.

E020-E050 are not yet reported results. Implemented CPU baselines and the
balanced-truncation primitive are infrastructure for those experiments, not a
substitute for fine-tuning.

## Input and truncation

Conversations are JSON lists of one to 36 turns. Each turn is formatted with
explicit prompt, response A, and response B markers.

Naive right truncation is forbidden for the main model because it can remove
response B more often and manufacture position bias. The main policy reserves
token budgets for prompt, A, and B, redistributes unused capacity, and keeps
both head and tail for long segments. Its tokenizer-independent allocation is
implemented in `src/pmldl_llm/truncation.py`. Track truncation rates for every
segment once it is connected to the neural tokenizer.

Initial comparison:

- maximum length 384 versus 512;
- naive right truncation versus balanced head-and-tail;
- no swap augmentation versus random swap;
- one inference order versus swap-averaged inference.

## Metrics

Primary:

- multiclass log loss.

Diagnostics:

- accuracy and macro-F1;
- per-class recall and confusion matrix;
- multiclass Brier score and calibration error;
- A/B symmetry error;
- log loss by class, length bucket, number of turns, language proxy, and
  truncation status;
- paired bootstrap confidence interval for model-to-model log-loss deltas.

Do not optimize a binary A-versus-B task and derive tie from a threshold.
Tie receives its own learned logit.

## Initial neural configuration

Reliable track:

- microsoft/deberta-v3-base, falling back to small if the 500-step runtime
  projection exceeds the budget;
- three-class cross-entropy;
- full fine-tuning, one to two epochs;
- backbone learning rate 1e-5 or 2e-5;
- weight decay 0.01, warmup ratio 0.06, gradient clipping 1.0;
- effective batch size 32 through gradient accumulation;
- dynamic padding and mixed precision;
- random A/B swap with probability 0.5;
- early stopping on selection-fold log loss.

Before a full run, benchmark 500 steps. If projected training is above seven
hours, reduce model size, maximum length, or epochs. Keep at least 1.5 hours
of safety margin.

## Reproducibility gates

Every experiment records:

- code version and config;
- dataset hashes and fold assignment;
- model and tokenizer names plus exact revisions;
- seed, hardware, package versions, duration, and peak memory;
- training curves, best checkpoint, OOF probabilities, and all metrics;
- input template and truncation statistics.

The final Kaggle notebook performs inference only, with internet disabled. It
must load attached weights and tokenizer, run schema checks, normalize
probabilities, and write a file named submission.csv.

## Definition of done for Phase 1

- At least two meaningful existing baselines are reproduced and attributed.
- One team-trained model has valid selection and final-holdout metrics.
- At least one architecture or hyperparameter improvement has an ablation.
- Results include uncertainty, calibration, symmetry, and error analysis.
- A clean offline run reproduces the reported metrics and submission.
- The anonymous report matches saved artifacts and contains all required
  sections from the course brief.
