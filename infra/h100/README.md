# deberta-peft-ablation launcher

This private launcher runs the managed DeBERTa fine-tuning ablation in PyDL on
one H100 80 GB. It compares full fine-tuning, LoRA and QLoRA under one frozen
split, seed and evaluation protocol. It requests the `alice-nlp-functions` pool
in `gpu_hainan_80g` with YT weight 2.

The input archives are kept outside Arcadia at
`/home/karimkhab/deberta_peft_ablation_inputs`. They contain no credentials.
ClearML is forced into offline mode inside the job; its offline cache is
returned in the PyDL `data` output.

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
