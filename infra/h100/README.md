# deberta-peft-ablation launcher

This private launcher runs the managed DeBERTa fine-tuning ablation in PyDL on
one H100 80 GB. It compares full fine-tuning, LoRA and QLoRA under one frozen
split, seed and evaluation protocol. It requests the `alice-nlp-functions` pool
in `gpu_hainan_80g` with YT weight 2.

The input archives are kept outside Arcadia at
`~/deberta_peft_ablation_inputs`. They contain no credentials.
ClearML is forced into offline mode inside the job; its offline cache is
returned in the PyDL `data` output.

## Bounded QLoRA HPO follow-up

`deberta_qlora_hpo_five` starts five independent one-GPU jobs concurrently.
Each job receives one pre-registered candidate from
`configs/experiments/E20260909170000000000.json`, evaluates fold 7 after
every epoch, keeps its best checkpoint, and stops after one epoch without a
meaningful improvement. It never reads folds 8 or 9 and does not query Kaggle.
The five jobs are deliberately independent: one H100 cannot make this
single-model training materially faster, whereas five candidates can be
compared in roughly the duration of the slowest candidate.

## Sprint 2 fixed multi-seed follow-up

`deberta_qlora_multiseed_three` starts three independent H100 jobs for seeds
42, 17 and 73. All non-seed settings are frozen to the HPO-winning
`high_lr_r16` candidate: rank 16, alpha 32, dropout 0.05, learning rate
2.8e-4, three training epochs and the same five-epoch scheduler horizon used
when that checkpoint was selected. Every job exports its adapter, tokenizer,
fold-7 original and swapped probabilities, a SHA256 manifest, and a canonical
`results/runs/` record. It does not read folds 8 or 9 and does not query Kaggle.

`deberta_qlora_multiseed_smoke` runs the seed-42 config on the balanced smoke
subset before the three full jobs are launched.

## Fold-8 calibration handoff

`deberta_qlora_seed42_fold8` is an inference-only job for the seed-42 adapter
selected on fold 7. Its source bundle must contain the verified adapter as
`qlora_adapter_seed42.zip` and its original manifest as
`seed42_model_manifest.json`. The job reads calibration fold 8, performs the
same original/swapped probability averaging as training-time evaluation, and
writes `id,target,winner_model_a,winner_model_b,winner_tie` plus an updated
SHA256 manifest under `artifacts/final_inputs/deberta_qlora_seed42/`. It never
selects a new checkpoint, retrains the model, or reads final holdout fold 9.

## Final seed-42 inference

After `configs/final_model.json` is frozen,
`deberta_qlora_seed42_final_inference` uses the same verified adapter once to
export raw DeBERTa probabilities for fold 9 and the Kaggle test set. It does
not retrain, tune a checkpoint, or alter the frozen ensemble weights and
temperature. The files are `fold9_predictions.csv` with labels for final
evaluation, `inference_predictions.csv` in Kaggle submission schema, and a
SHA256-bearing `final_model_manifest.json` under
`artifacts/final_inputs/deberta_qlora_seed42/`.

`code.tar.gz` holds `runner.py` and a git bundle of the tracked commit, plus a
`repo/` checkout of that same commit that is currently unused.

Dependencies are installed by `runner.py` from the pinned locks after it clones
the repository, not through the operation's `pip` parameter. That parameter
takes bare package specifiers and installs them one at a time; its installer
parses arguments with argparse and rejects `-r`, `-e` and other pip flags, so
requirement files cannot be passed through it.

Build and smoke-run from the Arcadia checkout:

```bash
ya make -r junk/karimkhab/deberta_peft_ablation
VH3_DEV=1 junk/karimkhab/deberta_peft_ablation/deberta_peft_ablation run \
  junk.karimkhab.deberta_peft_ablation.deberta_peft_ablation_smoke
```

Only after the smoke output is validated, run
`junk.karimkhab.deberta_peft_ablation.deberta_peft_ablation_full`.

## Why this lives in the repository

The launcher and `runner.py` produce the numbers reported for E2, so they are
versioned here rather than only in the Arcadia `junk/` directory and the input
archive. Copy `__init__.py`, `ya.make` and `conf.yaml` into
`junk/<login>/deberta_peft_ablation` in an Arcadia checkout to build it, and
ship `runner.py` inside `code.tar.gz` next to the repository bundle.

## Environment deviations recorded by every run

The job cannot reproduce the pinned locks exactly, and the run artifacts record
what actually ran:

- `transformers` is held at 4.56.2 rather than the locked 4.57.6. 4.57 walks
  every key of `model.state_dict()` while loading, which for a 4-bit model
  includes entries such as `...weight.absmax`; resolving those as module paths
  raises `AttributeError`. 4.55 is too old, as it predates the `dtype` keyword
  the notebook passes.
- `requirements-baseline.lock` is not applied. Its exact versions are absent
  from the internal mirror, and the CUDA image already provides that stack.
- `clearml` is not published on the internal mirror at all, so tracking reports
  status `unavailable` and only local evidence exists.
- The pinned checkpoint ships only `pytorch_model.bin`, which transformers
  refuses to load under the image's `2.6.0a0` torch. `runner.py` re-serialises
  it as safetensors; the revision and weights are unchanged.
