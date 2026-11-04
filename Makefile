PY ?= python3
export PYTHONPATH := src:.

.PHONY: test bench report clean colab-build colab-validate colab-review
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
