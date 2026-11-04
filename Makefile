PY ?= python3
export PYTHONPATH := src:.

.PHONY: test bench report clean
test:
	$(PY) -m pytest

# Each script reads its thresholds from gates.toml and writes results/<name>.json.
bench:
	@for f in bench/phase*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

report:
	$(PY) eval/report.py

clean:
	rm -rf .pytest_cache .hypothesis __pycache__
