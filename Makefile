PYTHON ?= python3
export PYTHONPATH := src

.PHONY: new-experiment prepare-full submit-experiment
.PHONY: freeze-folds audit baseline baseline-run sparse-baseline blend-baselines
.PHONY: check-results validate-run collect-results verify-artifacts test

new-experiment:
	$(PYTHON) scripts/team_experiment.py start

prepare-full:
	$(PYTHON) scripts/team_experiment.py prepare-full $(EXPERIMENT)

submit-experiment:
	$(PYTHON) scripts/team_experiment.py submit $(EXPERIMENT)

freeze-folds:
	$(PYTHON) scripts/freeze_folds.py

audit:
	$(PYTHON) scripts/audit_data.py

baseline:
	$(PYTHON) scripts/train_bias_baseline.py

baseline-run:
	$(PYTHON) scripts/run_bias_baseline_experiment.py

sparse-baseline:
	$(PYTHON) scripts/train_sparse_baseline.py

blend-baselines:
	$(PYTHON) scripts/blend_baselines.py

check-results:
	$(PYTHON) scripts/check_results.py

validate-run:
	$(PYTHON) scripts/validate_run.py $(RUN_DIR)

collect-results:
	$(PYTHON) scripts/collect_results.py

verify-artifacts:
	$(PYTHON) scripts/verify_artifacts.py

test:
	$(PYTHON) -m unittest discover -s tests -v
