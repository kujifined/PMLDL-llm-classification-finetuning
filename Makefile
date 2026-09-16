PYTHON ?= python
export PYTHONPATH := src

.PHONY: audit freeze-folds structural-baseline sparse-baseline sparse-sweep
.PHONY: train-deberta calibrate finalize bundle test release-preflight

freeze-folds:
	$(PYTHON) scripts/freeze_folds.py

audit:
	$(PYTHON) scripts/audit_data.py

structural-baseline:
	$(PYTHON) scripts/train_bias_baseline.py

sparse-baseline:
	$(PYTHON) scripts/train_sparse_baseline.py

sparse-sweep:
	$(PYTHON) scripts/sweep_sparse_sprint2.py --config configs/experiments/E202609140001.json

train-deberta:
	$(PYTHON) scripts/train_e2_qlora.py --config configs/experiments/E20260914010000000000.json

calibrate:
	$(PYTHON) scripts/calibrate_and_blend.py --stage select --config configs/final_model_search.json --output-dir results/final
	$(PYTHON) scripts/calibrate_and_blend.py --stage blend --config configs/final_model_search.json --output-dir results/final

finalize:
	$(PYTHON) scripts/finalize_release.py $(FINALIZE_ARGS)

bundle:
	$(PYTHON) scripts/build_kaggle_bundle.py

test:
	$(PYTHON) -m unittest discover -s tests -v

release-preflight: test
	$(PYTHON) scripts/check_release.py
