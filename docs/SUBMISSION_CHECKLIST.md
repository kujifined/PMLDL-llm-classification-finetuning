# Stage 2 submission checklist

This checklist mirrors the 2026 course brief. A checked box must point to a
reproducible artifact, not merely to planned work.

## Packaging

- [ ] `project.zip` and every file inside it are anonymized: no names, emails,
  usernames, home-directory paths, notebook metadata, repository remotes, or
  other personal identifiers.
- [ ] The archive contains source code and an anonymized `project.pdf`.
- [ ] Every borrowed code fragment, text passage, figure, model, dataset, and
  public notebook is linked to its original source and mapped to the affected
  local component in `docs/SOURCES.md`.
- [ ] `make audit` passes after the official competition data is placed in the
  documented location.

## Three minimal ML requirements (50 points)

- [ ] Several existing solutions have been reproduced on the same frozen split
  and their metrics are reported.
- [ ] The team has trained or fine-tuned a new model for this task and reported
  its metrics.
- [ ] The proposed model has been modified through architecture or
  hyperparameter changes, with a controlled metric comparison.

Missing one requirement can be graded Major; missing two or more can be graded
Severe. The current CPU baselines alone do not complete this section.

## Implementation and results (25 points)

- [ ] Data checksum, schema, label-order, null-turn, duplicate-group, and A/B
  swap invariants pass.
- [ ] Every compared model uses the same frozen selection examples and metric.
- [ ] Fold 8 is calibration-only and fold 9 is opened once after model choice.
- [ ] Report tables are generated from or checked against saved metrics.
- [ ] Saved checkpoints reproduce saved probabilities.
- [ ] Results include uncertainty and honest selection-bias caveats.
- [ ] Error analysis covers class, length, turn count, truncation, and language
  proxy strata.

## Reproducibility (15 points)

- [ ] A private Git commit identifies the exact submitted code.
- [ ] Python/package versions, config, seeds, hardware, runtime, and peak memory
  are recorded.
- [ ] Data acquisition and exact dataset hashes are documented.
- [ ] Pretrained model/tokenizer repository, immutable revision, license, and
  offline Kaggle input are recorded.
- [ ] One clean-environment run reproduces the claimed metrics.
- [ ] One internet-disabled Kaggle run creates a valid hidden-test
  `submission.csv` within the nine-hour limit.

## Required `project.pdf` sections (10 points)

- [ ] Problem statement: task, data source, and evaluation metric.
- [ ] Baselines: existing solutions and their metrics.
- [ ] Proposed model: architecture, fine-tuning procedure, and metrics.
- [ ] Model improvements: architecture/hyperparameter changes and metric effect.
- [ ] Results: common-protocol comparison, tables, plots, and analysis.
- [ ] Reproducibility: environment, dependencies, data access, and exact steps.
- [ ] Conclusion: findings and limitations.
- [ ] All claims, numbers, captions, and references agree with code/artifacts.

## Local data check

~~~bash
make audit
~~~

Run `make audit` separately after placing the official data archive in the
location from `data/README.md`. Before packaging, perform an anonymity scan
for names, emails, usernames, local paths, Git metadata, and notebook kernels.
