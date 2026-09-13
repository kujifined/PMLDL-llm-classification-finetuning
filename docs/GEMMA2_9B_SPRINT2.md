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

## Deliverables

The tracked repository receives the immutable config, clean notebook, aggregate run metadata, metrics, and source attribution. Private run artifacts contain each completed arm's best adapter, training curve, fold-7 probabilities, environment record, study record, and `model_manifest.json` with SHA256 checksums. Fold-8 probabilities are generated only after the fold-7 candidate is accepted for calibration.
