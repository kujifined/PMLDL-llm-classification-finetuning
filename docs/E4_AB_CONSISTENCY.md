# E4: QLoRA A/B consistency loss

## Question

Can an explicit order-consistency objective reduce A/B position sensitivity without materially degrading preference log loss?

## Controlled comparison

The control is the QLoRA arm of E2 (`E20260905212645934620`): DeBERTa-v3-base, revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, seed 42, frozen prompt-group split, three epochs, effective batch size 32, learning rate `2e-4`, maximum length 512, balanced head/tail truncation, and deterministic 50% random A/B swap.

E4 changes one training factor. Cross-entropy is computed on the same random view as E2. A second forward example uses the opposite response order; after mapping its logits back to the primary label order, E4 adds `0.1 * Jensen-Shannon divergence` between the two probability vectors.

The dense paired objective was too slow on a Kaggle T4. The production E4 run therefore uses a deterministic stride-16 stochastic estimator: one of every 16 shuffled microbatches receives the opposite-order forward pass and its JS term is multiplied by 16. The mean estimator weight over a stride is exactly one, so this estimates the same dense consistency objective without bias while retaining E2's primary-example microbatch geometry. Calibration and final-holdout folds remain unopened.

## Run protocol

1. Run the committed Kaggle launcher with GPU and Internet enabled. The launcher clones an exact public Git commit, retrieves the two ClearML keys from Kaggle Secrets without printing them, verifies competition file checksums, installs the pinned environment, and executes the managed notebook.
2. The first version is a smoke run: 72 balanced training rows, 36 balanced selection rows, 128 tokens, and three microbatches. It validates the entire GPU, 4-bit, ClearML, evaluation, checkpoint, and artifact path.
3. Import the smoke output into the repository and run `make prepare-full EXPERIMENT=E20260909125807233308`. This is the only supported transition to `smoke_test=false`.
4. Commit and push that transition, pin a new Kaggle launcher version to the new commit, and run the full experiment.

## Decision rule

The fixed E2 control is selection log loss `1.03428689372673` and raw swap L1 error `0.16203328866204142`. E4 is promising if raw swap error improves and log loss is no worse by more than `0.01`; it is clearly better if both metrics improve. Smoke metrics are diagnostic only and must never be compared with the full E2 result.

The runtime guard aborts a full run when the step-20 projection exceeds 27,000 seconds. Static or smoke validation is not evidence that full training succeeded.

## Run attempts

- Kaggle Version #1, `E4 smoke pinned ea1924a`: successful end-to-end smoke on T4 x2; ClearML task `4941fccb9eed4710bacfdf0aa51f753d`.
- Kaggle Version #2, `E4 full pinned c3525b9`: intentionally stopped by the runtime guard after 20 optimizer steps. The dense counterpart pass projected `46,205` seconds versus the `27,000`-second safety limit; ClearML task `a8b0a69ed2894a1c9d4d358079dba393`. No model-selection result was produced and this attempt must not be compared with E2.
- Kaggle Version #3, `E4 full sparse-JS pinned c3c443f`: completed successfully in `28,431.629` seconds (`7h 53m 52s` measured inside the run; Kaggle reported `7h 58m 44s`). Run `E20260909125807233308__s42__c3c443f8__20260909T150336809694Z`; ClearML task `985c621c24d34872948935e61b640273`.

## Results and decision

The completed full E4 run is compared only with its preregistered E2 QLoRA control on the same selection fold.

| Model | Selection log loss | Raw swap L1 error | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| E2 QLoRA control | 1.0342869 | 0.1620333 | 0.4617125 | 0.4589996 |
| E4 sparse-JS | 1.0357633 | 0.1731375 | 0.4624086 | 0.4595449 |
| E4 minus E2 | +0.0014764 | +0.0111043 | +0.0006961 | +0.0005453 |

E4 satisfies the log-loss non-inferiority allowance (`+0.00148 < +0.01`) but fails the primary consistency condition: raw swap error worsened by `0.01110` (about `6.85%`) instead of improving. The experiment is therefore **rejected as a model-selection candidate**. E2 QLoRA remains the selected arm, and no Kaggle submission should be created from E4.

The calibration and final-holdout folds remain unopened. The repository contains only the immutable config snapshot, run provenance, and aggregate metrics. The checkpoint, full training history, and row-level validation predictions remain in the private Kaggle/ClearML artifacts and are intentionally not committed to the public repository.
