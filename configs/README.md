# Configuration

- `project.json` declares the dataset, frozen folds, metric, and final-release
  evaluation roles.
- `split.json` defines the fixed train, selection, calibration, and final
  holdout folds.
- `experiments/` contains the retained baseline, sparse sweep, QLoRA search,
  and selected DeBERTa configurations. File names describe the experiment;
  immutable experiment IDs remain inside JSON for provenance.
- `final_model_search.json` records the two candidates and the calibration
  procedure.
- `final_model.json` is the immutable final ensemble configuration.

The JSON schemas describe the configuration and metrics formats. No API keys,
personal paths, or runtime credentials are required in this directory.
