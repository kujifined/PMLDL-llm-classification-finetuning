PYTHON ?= python
export PYTHONPATH := src

.PHONY: audit freeze-folds structural-baseline sparse-baseline sparse-sweep
.PHONY: qlora-hpo train-deberta calibrate finalize bundle

freeze-folds:
	$(PYTHON) scripts/freeze_folds.py

audit:
	$(PYTHON) scripts/audit_data.py

structural-baseline:
	$(PYTHON) scripts/train_bias_baseline.py

sparse-baseline:
	$(PYTHON) scripts/train_sparse_baseline.py

sparse-sweep:
	$(PYTHON) scripts/sweep_sparse_sprint2.py --config configs/experiments/tfidf_sprint2_sweep.json

qlora-hpo:
	$(PYTHON) scripts/run_qlora_hpo_trial.py --config configs/experiments/qlora_hyperparameter_search.json --trial-id high_lr_r16

train-deberta:
	$(PYTHON) scripts/train_e2_qlora.py --config configs/experiments/deberta_qlora_seed42.json

calibrate:
	$(PYTHON) scripts/calibrate_and_blend.py --stage select --config configs/final_model_search.json --output-dir results/final
	$(PYTHON) scripts/calibrate_and_blend.py --stage blend --config configs/final_model_search.json --output-dir results/final

finalize:
	$(PYTHON) scripts/finalize_release.py $(FINALIZE_ARGS)

bundle:
	$(PYTHON) scripts/build_kaggle_bundle.py
