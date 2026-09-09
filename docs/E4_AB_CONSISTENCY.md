# E4: QLoRA A/B consistency loss

## Question

Can an explicit order-consistency objective reduce A/B position sensitivity without materially degrading preference log loss?

## Controlled comparison

The control is the QLoRA arm of E2 (`E20260905212645934620`): DeBERTa-v3-base, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, seed 42, frozen prompt-group split, three epochs, effective batch size 32, learning rate `2e-4`, maximum length 512, balanced head/tail truncation, and deterministic 50% random A/B swap.

E4 changes one training factor. Cross-entropy is computed on the same random view as E2. A second forward example uses the opposite response order; after mapping its logits back to the primary label order, E4 adds `0.1 * Jensen-Shannon divergence` between the two probability vectors.

The paired batch is sized so that the number of sequences processed per forward pass matches the E2 QLoRA arm on both T4 and H100 paths. Calibration and final-holdout folds remain unopened.

## Run protocol

1. Run the committed Kaggle launcher with GPU and Internet enabled. The launcher clones an exact public Git commit, retrieves the two ClearML keys from Kaggle Secrets without printing them, verifies competition file checksums, installs the pinned environment, and executes the managed notebook.
2. The first version is a smoke run: 72 balanced training rows, 36 balanced selection rows, 128 tokens, and three microbatches. It validates the entire GPU, 4-bit, ClearML, evaluation, checkpoint, and artifact path.
3. Import the smoke output into the repository and run `make prepare-full EXPERIMENT=E20260909125807233308`. This is the only supported transition to `smoke_test=false`.
4. Commit and push that transition, pin a new Kaggle launcher version to the new commit, and run the full experiment.

## Decision rule

The fixed E2 control is selection log loss `1.03428689372673` and raw swap L1 error `0.16203328866204142`. E4 is promising if raw swap error improves and log loss is no worse by more than `0.01`; it is clearly better if both metrics improve. Smoke metrics are diagnostic only and must never be compared with the full E2 result.

The runtime guard aborts a full run when the step-20 projection exceeds 27,000 seconds. Static or smoke validation is not evidence that full training succeeded.
