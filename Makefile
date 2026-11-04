PY ?= python3
export PYTHONPATH := src:.

.PHONY: test bench report clean colab-build colab-validate colab-review pdm-fetch pdm-study pdm-gates pdm-report pdm-figures pdm-posthoc
test:
	$(PY) -m pytest

# Each script reads its thresholds from gates.toml and writes results/<name>.json.
bench:
	@for f in bench/phase*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

report:
	$(PY) eval/report.py

clean:
	rm -rf .pytest_cache .hypothesis __pycache__

# ---- Colab notebook (colab/): sources in colab/src/*.py -> colab/NeuroStore_Colab_Benchmarks_v2.ipynb
# colab-validate and colab-review need jax, nbclient, ipykernel, pandas, matplotlib in $(COLAB_PY); the static tests in `make test` do not.
COLAB_PY ?= $(PY)
colab-build:
	$(PY) colab/build_notebook.py

colab-validate: colab-build
	$(COLAB_PY) colab/validate_v2.py --runs smoke multidev

colab-review:
	$(COLAB_PY) colab/review_experiments.py
	$(PY) colab/make_review.py

# ---- nsharness + PdM study (studies/pdm_cmapss/): protocol and gates are pre-registered; results land in results/pdm_cmapss/
# The scripts are run with `-I` (isolated: they read downloaded data) and put src/ on the path themselves.
pdm-fetch:
	$(PY) -I studies/pdm_cmapss/fetch_data.py

pdm-study:
	@for s in validate tune tune_others test sweeps completion ood timing; do $(PY) -I studies/pdm_cmapss/run_study.py --stage $$s || exit 1; done

pdm-gates:
	$(PY) -I studies/pdm_cmapss/evaluate_gates.py

pdm-report:
	$(PY) -I studies/pdm_cmapss/make_report.py

# figures need matplotlib (optional; the committed PNGs are used when it is missing): FIG_PY=/path/to/python-with-matplotlib make pdm-figures
FIG_PY ?= $(PY)
pdm-figures:
	$(FIG_PY) studies/pdm_cmapss/make_figures.py

# exploratory control run AFTER the pre-registered study (needs results/pdm_cmapss/sweeps_cache from `make pdm-study`)
pdm-posthoc:
	$(PY) -I studies/pdm_cmapss/posthoc_quantized.py
