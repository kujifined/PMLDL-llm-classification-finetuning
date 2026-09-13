# Sprint 2: Gemma-2 9B LoRA versus QLoRA

## Question

Does the larger Gemma-2-9B-IT backbone produce a stronger selection-fold candidate than the E2 DeBERTa QLoRA control, and how do FP16/BF16 LoRA and 4-bit QLoRA change across one, two, and three epochs?

## Frozen comparison

- Parent: E2 QLoRA run `E20260905212645934620__s42__38aea1ac__20260907T052625009035Z`.
- Data: official competition train set with the existing prompt-group split.
- Train: folds 0-6; model selection: fold 7.
- Fold 8 calibration and fold 9 final holdout remain unopened.
- Features: prompt, response A, and response B only. Model identities are excluded.
- Preprocessing: balanced head-tail truncation to 512 tokens with budget weights 1:2:2.
- Augmentation and inference: deterministic 50% random A/B swap during training and swap-averaged probabilities during evaluation.
- Arms: FP16/BF16 LoRA and 4-bit NF4 QLoRA with identical rank, alpha, dropout, target modules, optimiser, effective batch size, and learning-rate schedule.
- Checkpoints: selection metrics are recorded after epochs 1, 2, and 3; the best epoch within each arm is retained.

The preregistered E2 control is selection log loss `1.03428689372673` and raw swap L1 error `0.16203328866204142`. Gemma becomes a downstream candidate only if its fold-7 log loss is competitive with or better than this control. Kaggle public leaderboard is not used for model or epoch selection.

## Resource policy

The notebook computes an activation-free lower bound for full AdamW fine-tuning before loading the model. For a 9B model, FP16 weights and gradients plus FP32 master weights and two Adam moments require about 134 GiB before activations and allocator overhead, so two 16 GiB T4 GPUs cannot pass the gate. Full fine-tuning is therefore not attempted on Kaggle T4.

FP16/BF16 LoRA may use model sharding across two GPUs. If it still raises CUDA OOM, the run records `skipped_resource_limit` and proceeds to QLoRA. QLoRA uses NF4 with double quantization and keeps the classification head unquantized. A runtime projection guard stops an arm that cannot finish inside the configured Kaggle budget.

## Executed T4x2 preflight

The full protocol was launched on two Tesla T4 GPUs from implementation commit
`af25318db9a06344a52ebb88cec5918c1154adc1`. The run used all 40,235 training
rows from folds 0-6 and kept all 5,746 fold-7 rows reserved for selection.
Folds 8 and 9 were not opened.

| Arm | Four measured micro-batches | Projected three-epoch runtime | Versus 11-hour gate | Outcome |
| --- | ---: | ---: | ---: | --- |
| LoRA | 34.59 s | 289.96 h / 12.08 days | 26.36x | stopped by runtime guard |
| QLoRA | 39.06 s | 327.41 h / 13.64 days | 29.76x | stopped by runtime guard |

Full AdamW fine-tuning was also rejected before model loading: its
activation-free memory lower bound was 134.11 GiB, versus 29.12 GiB physically
available and 26.21 GiB allowed by the safety margin.

These runtimes are early linear extrapolations, not measured full-run
durations. They are sufficient for the configured resource gate because both
arms exceed the Kaggle limit by more than 26x. Neither arm reached an epoch
checkpoint, so no fold-7 probabilities or comparable validation metrics exist.
The smoke run remains an engineering check only and is not model-selection
evidence.

The scientific outcome is therefore **resource-limited, no comparison**.
Gemma-2 9B cannot replace the E2 control on this evidence; the selected model
remains E2 DeBERTa-v3-base QLoRA. Running Gemma further requires either larger
compute or a separately preregistered constrained protocol. Silently reducing
the dataset, context length, or epoch schedule would answer a different
question.

Machine-readable evidence is stored in
`results/preflights/E20260913115603636696__t4x2.json` and the canonical live
tracking record is ClearML task
[`c7b664ae38aa412bac32a5717f09dc2b`](https://app.clear.ml/projects/a824a1f17fa3493da105af69140f8c0d/tasks/c7b664ae38aa412bac32a5717f09dc2b/artifacts).

## Deliverables

The tracked repository receives the immutable config, clean notebook, aggregate run metadata, metrics, and source attribution. Private run artifacts contain each completed arm's best adapter, training curve, fold-7 probabilities, environment record, study record, and `model_manifest.json` with SHA256 checksums. Fold-8 probabilities are generated only after the fold-7 candidate is accepted for calibration.
