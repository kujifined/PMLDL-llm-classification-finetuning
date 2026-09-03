# Contributing

This is a six-person course-team repository. Keep all development compatible
with the frozen evaluation protocol and the anonymized Stage 2 submission.

## Before opening a pull request

~~~bash
make test
make audit
make check-results
make verify-artifacts
~~~

Use a focused branch and describe the experiment ID, config change, validation
fold, runtime, and resulting log loss in the pull request. Do not merge metric
claims unless their aggregate `metrics.json` and source attribution are updated.

## Non-negotiable rules

- Never commit raw Kaggle CSV/ZIP files, model checkpoints, API tokens, or local
  environment files.
- Never use `model_a` or `model_b` as model inputs: they are absent from the
  hidden test schema.
- Never regenerate `data/splits/folds.csv` during model development.
- Never use fold 8 for model selection or fold 9 before the shortlist is frozen.
- Attribute every borrowed notebook, code fragment, pretrained model, dataset,
  statement, and figure in `docs/SOURCES.md` before merging it.
- Keep names, emails, usernames, absolute home paths, Git metadata, and notebook
  author metadata out of the final `project.zip` and `project.pdf`.

Heavy artifacts should be shared through a private team-only storage location
and referenced by checksum/revision. The Git repository keeps only source,
configuration, documentation, frozen split metadata, and small aggregate
metrics.
