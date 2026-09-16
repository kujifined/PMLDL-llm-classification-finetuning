# Project status and remaining course gates

This page separates verified completion from the few external gates that are
not under the repository's control.

## Sprint 1: complete locally

- Competition data contract, checksum audit, duplicate-prompt analysis, and
  null-turn handling are implemented.
- The immutable prompt-grouped split keeps folds 0-6 for training, fold 7 for
  selection, fold 8 for calibration, and fold 9 for one final holdout.
- Uniform, class-prior, structural, sparse TF-IDF, and exploratory blend
  baselines have saved metrics, probabilities, and reconstruction checks.
- A/B swap augmentation and exactly swap-symmetric inference are implemented.
- Balanced head-and-tail truncation has exhaustive small-case tests.
- E2 compares full fine-tuning, LoRA, and QLoRA on `deberta-v3-base` with the
  same split, seed, maximum length, optimizer-step budget, A/B swap training,
  and swap-averaged evaluation. At three epochs, QLoRA is best at **1.03429**
  validation log loss, LoRA reaches 1.04147, and full fine-tuning degrades to
  1.08737.
- The final inference notebook is Internet-Off and test-size independent. Its
  clean Kaggle execution created a schema-valid `submission.csv`; the fixed
  58% QLoRA / 42% sparse blend first received 1.02067 public log loss. A
  separate five-candidate, fold-7-only QLoRA follow-up selected rank 16, LR
  2.8e-4 at epoch three (1.01627 local log loss); its unchanged 58/42 blend
  received **1.01108** public log loss.
- `make test`, `make check-results`, `make collect-results`,
  `make verify-artifacts`, and `validate_run` for the one-, two-, and
  three-epoch E2 records passed before submission.
- First neural result: E060, an exact swap-equivariant pair encoder
  (`microsoft/deberta-v3-small`, 1 epoch), log loss 1.050053 on fold 7 --
  beats the structural baseline (E002) and statistically ties the sparse
  TF-IDF baseline (E010/E011). Reported informally in `docs/RESULTS.md`
  because it was run from a dirty Git tree, so `run_policy.require_clean_git`
  keeps it out of `results/leaderboard.csv` for now; see the reproducibility
  gap note below.

E6 is deliberately not included. It is an optional additional task; Sprint 1
priority is the required baselines and main experiments.

## Sprint 2: Karim's fold-8 handoff is complete

- The fixed HPO winner was evaluated at epoch four. Fold-7 log loss worsened
  from 1.01627 at epoch three to 1.01883, so the written stopping rule excluded
  epoch five.
- The epoch-three candidate was trained cleanly at seeds 42, 17 and 73. The
  individual fold-7 log losses are 1.01505, 1.02800 and 1.02807.
- Their probability mean reaches 1.01889. It improves raw A/B symmetry but does
  not beat seed 42, so the recommended DeBERTa candidate is seed 42.
- All three adapter archives include tokenizer files. Their manifests record
  class order, source commit, run identity and SHA256; all three 5,746-row
  fold-7 probability files are retained for team comparison.
- The runs came from clean commit `e76c1fcb` in [one three-job H100 Nirvana
  process](https://nirvana.yandex-team.ru/process/d5e9ed94-7877-4570-814d-285ee16ca215).
  The H100 image lacked ClearML, so the source runs record `unavailable` and a
  separate [historical-import task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/b158f2d9a9394b68bd7188459e0b375e/general)
  holds the retained curves, configs, adapters, manifests and predictions.
- After the team shortlist retained seed 42, an inference-only H100 run created
  5,748 fold-8 rows with the requested five-column schema. The file SHA256 is
  `f388b1e93444f1ada9cc345f3135f15ea0924979f43071e8156eb1aa60ed5115`;
  the unchanged source adapter SHA256 is
  `d2bebee99d24417aa26435d1126f20a30ffeb41466db19ea20dca64177ead4a6`.
  The [Nirvana process](https://nirvana.yandex-team.ru/process/d1fcd38d-e1c3-47ef-9266-19773522a91a)
  and [ClearML handoff task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/0edd517d647341f0a3e9d6a1a733e1ae/general)
  preserve execution and artifact provenance.

The next action belongs to the team calibration stage: Arseny combines this
file with the sparse and symmetric fold-8 predictions, selects the bounded
blend and temperature, and freezes `final_model.json`. Fold 9 remains closed
until that final solution is frozen.

## Explicit limitations

- Kaggle's public score is useful external evidence, but it is not used to
  tune the blend or select another local model. Fold 8 is now used only for
  calibration after the shortlist decision; fold 9 remains unopened.
- The H100 environment used `transformers` 4.56.2 rather than the pinned
  4.57.6 and re-serialised the pinned model weights to safetensors. The run
  metadata and `infra/h100/README.md` record these deviations.
- The H100 mirror could not initialise ClearML (`tracking_status=unavailable`).
  The shared project now contains a dedicated **E2 historical import** task
  with the run identity, protocol, final metrics, and Kaggle result. The
  reproducible importer is `scripts/import_e2_to_clearml.py`; it uploaded the
  complete 2,695-point history and `config.json`, `metrics.json`, and
  `run.json` into [the task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/77b645cca3cd46f08f2cdff46eb0c179/general).
  This record must remain labelled as a historical import, never as live
  tracking of the original H100 job.
- The five completed H100 QLoRA follow-up trials are recorded separately as a
  historical import. Their versioned source is
  `docs/evidence/E20260909170000000000_hpo_summary.json`; the reproducible
  importer is `scripts/import_hpo_to_clearml.py`, which created [the HPO
  task](https://app.clear.ml/projects/ab1057f78e8b4cafae8460dffdc3f609/tasks/5539883a8cb84f1cb5e0f923de07b24c/general).
  This preserves the fact that ClearML was connected after the H100 execution
  rather than during it.
- E060 remains an informal result from a dirty tree. Its Sprint 2 owner must
  reproduce it from a clean commit before Arseny's fold-7 comparison can treat
  it as a tracked candidate.

## Remaining external actions

1. Obtain green pull-request CI for the Sprint 2 E2 branch and pass the
   validated fold-8 predictions and manifest to Arseny.
2. Wait for the team calibration result; do not refit or open fold 9 at this
   stage.
3. After the team freezes one final ensemble, evaluate fold 9 exactly once and
   let the release owner assemble the anonymous `project.zip` and `project.pdf`.

`docs/SPRINT_1_E2_REPORT.md` is the concise submission-ready account of E2;
`docs/SPRINT_2_E2_MULTI_SEED_REPORT.md` is the Sprint 2 handoff; and
`docs/RESULTS.md` remains the detailed evidence ledger.
