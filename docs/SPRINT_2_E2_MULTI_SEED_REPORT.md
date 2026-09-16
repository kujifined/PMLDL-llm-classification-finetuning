# Sprint 2: E2 QLoRA epoch and seed stability

## Assignment and outcome

Karim's Sprint 2 task is to keep the winning DeBERTa QLoRA configuration
fixed, check whether training should continue beyond epoch three, train two
additional seeds at the selected epoch, and compare the three individual
models with their probability mean on fold 7.

The task is complete through the fold-7 handoff gate. Epoch four made the
winning seed-42 model worse, from **1.01627** to **1.01883** log loss, so the
predefined rule did not permit epoch five. Three clean epoch-three H100 runs
then produced a best single model of **1.01505** at seed 42. The three-seed
mean reached **1.01889**, so it is retained as an evaluated alternative while
seed 42 is the candidate passed to ensemble selection.

The team shortlist retained seed 42, so its fold-8 probabilities were generated
after that decision for calibration and bounded ensemble selection. Fold 9
remains unopened. Kaggle was not used for the fold-7 selection.

## Fixed protocol

- backbone: `microsoft/deberta-v3-base`, revision
  `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`;
- train folds: 0-6; selection fold: 7;
- QLoRA: rank 16, alpha 32, dropout 0.05, NF4 with double quantisation;
- learning rate: `2.8e-4`; scheduler horizon: five epochs;
- selected training length: three epochs;
- AdamW, weight decay 0.01, warmup ratio 0.06, maximum gradient norm 1.0;
- maximum length 512, effective batch size 32;
- random A/B swap probability 0.5 during training and swap-averaged inference;
- seeds fixed before execution: 42, 17 and 73.

The fourth-epoch evidence comes from the already completed bounded HPO study,
which ran the winning `high_lr_r16` trajectory through epoch four under this
same protocol. Repeating that comparison would spend another H100 run without
changing the decision.

## Epoch stopping decision

| Epoch | Fold-7 log loss | Decision |
|---:|---:|---|
| 1 | 1.08153 | continue |
| 2 | 1.03445 | continue |
| 3 | **1.01627** | provisional best |
| 4 | 1.01883 | stop; no epoch 5 |

The fourth epoch is worse by 0.00256. This satisfies the explicit stopping
rule directly: epoch five is allowed only after an improvement at epoch four.

## Multi-seed comparison

| Candidate | Fold-7 log loss | Accuracy | Macro-F1 | ECE-15 | Brier | Raw swap L1 |
|---|---:|---:|---:|---:|---:|---:|
| seed 42 | **1.01505** | **0.48747** | **0.48420** | 0.01583 | **0.60864** | 0.16376 |
| mean of seeds 17, 42, 73 | 1.01889 | 0.48016 | 0.48001 | **0.01547** | 0.61150 | **0.13605** |
| seed 17 | 1.02800 | 0.46711 | 0.46777 | 0.03340 | 0.61816 | 0.19701 |
| seed 73 | 1.02807 | 0.47355 | 0.47328 | 0.01775 | 0.61781 | 0.16072 |

The individual seed mean is 1.02371. Probability averaging reduces raw A/B
asymmetry and protects against the two weaker seeds, but it is 0.00384 worse
than seed 42 on the selection metric. The handoff recommendation is therefore
the seed-42 adapter, not the three-seed mean.

The clean seed-42 rerun is 0.00122 better than the earlier HPO measurement
(1.01505 versus 1.01627). Both use the same fixed configuration. This small
difference is recorded as run-to-run training variation; it does not trigger
another search.

## Reproducibility and handoff

The three independent jobs ran from clean commit
`e76c1fcb6578542a18d426126a521cd91b4d6713` on one H100 80 GB each in the
[Nirvana process](https://nirvana.yandex-team.ru/process/d5e9ed94-7877-4570-814d-285ee16ca215).
They took 983-989 seconds each and used 4,250 MB peak GPU memory as reported by
the runner.

Versioned evidence:

- `configs/experiments/E20260914010000000000.json` - seed 42;
- `configs/experiments/E20260914010000000001.json` - seed 17;
- `configs/experiments/E20260914010000000002.json` - seed 73;
- `results/runs/<run_id>/` - exact config, run metadata, epoch history and
  final metrics for each seed;
- `results/e2_multiseed/fold7_comparison.csv` - individual and averaged
  fold-7 comparison;
- `results/e2_multiseed/summary.json` - selected candidate, hashes and closed
  evaluation boundaries at selection time;
- `scripts/train_e2_qlora.py` and `scripts/compare_e2_multiseed.py` - training,
  artifact validation and comparison code.
- `scripts/infer_e2_qlora_fold8.py` - inference-only fold-8 handoff from the
  selected adapter; it never reads fold 9.
- `results/e2_multiseed/fold8_handoff.json` - row count, hashes, run lineage,
  Nirvana process and ClearML task for the delivered calibration file.

Ignored handoff artifacts are present under `artifacts/<run_id>/`. Each seed
directory contains the QLoRA adapter and tokenizer in
`models/qlora_adapter.zip`, a SHA256-bearing `model_manifest.json`, and the
5,746-row `predictions/fold7_predictions.csv`. The averaged probabilities and
their manifest are under `artifacts/e2_multiseed/`.

ClearML was unavailable inside the H100 image, and every original `run.json`
records that fact. The retained histories and handoff artifacts were uploaded
after completion to a separate [historical-import task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/b158f2d9a9394b68bd7188459e0b375e/general).
The import is reproducible with `scripts/import_multiseed_to_clearml.py` and is
not represented as live tracking of the original jobs.

## Fold-8 calibration handoff

After Arseny's fold-7 shortlist retained DeBERTa, the frozen seed-42 adapter was
used for inference only in [this H100 Nirvana process](https://nirvana.yandex-team.ru/process/d1fcd38d-e1c3-47ef-9266-19773522a91a).
The resulting `fold8_predictions.csv` has 5,748 unique rows and the exact
handoff schema `id,target,winner_model_a,winner_model_b,winner_tie`. Its SHA256
is `f388b1e93444f1ada9cc345f3135f15ea0924979f43071e8156eb1aa60ed5115`.
The source adapter SHA256 remained
`d2bebee99d24417aa26435d1126f20a30ffeb41466db19ea20dca64177ead4a6`.

The predictions and updated manifest are available in the shared
[ClearML handoff task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/0edd517d647341f0a3e9d6a1a733e1ae/general).
Local validation confirmed the frozen fold-8 IDs and targets, finite
non-negative probabilities summing to one, zero overlap with fold 9, and both
recorded hashes. Fold 9 remains closed until the final ensemble and calibration
are frozen.
