PYTHON ?= python3
export PYTHONPATH := src

.PHONY: freeze-folds audit baseline sparse-baseline blend-baselines check-results verify-artifacts test

freeze-folds:
	$(PYTHON) scripts/freeze_folds.py

audit:
	$(PYTHON) scripts/audit_data.py

baseline:
	$(PYTHON) scripts/train_bias_baseline.py

sparse-baseline:
	$(PYTHON) scripts/train_sparse_baseline.py

blend-baselines:
	$(PYTHON) scripts/blend_baselines.py

check-results:
	$(PYTHON) scripts/check_results.py

verify-artifacts:
	$(PYTHON) scripts/verify_artifacts.py

test:
	$(PYTHON) -m unittest discover -s tests -v
