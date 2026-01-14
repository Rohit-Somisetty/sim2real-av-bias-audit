PYTHON ?= python
PIP ?= pip
PACKAGE := sim2real
OUTDIR := outputs

.PHONY: install lint test data analyze clean

install:
	$(PIP) install -e .[dev]

lint:
	ruff check src tests
	ruff format --check src tests
	ruff format src tests
	ruff check --fix src tests
	ruff format src tests

pytest:
	pytest -q

test:
	pytest -q

clean:
	rm -rf $(OUTDIR) .pytest_cache .ruff_cache build dist *.egg-info

analyze:
	$(PYTHON) -m $(PACKAGE).cli analyze --data $(OUTDIR)/data.parquet --outdir $(OUTDIR)

data:
	$(PYTHON) -m $(PACKAGE).cli generate-data --out $(OUTDIR)/data.parquet --trips 200 --seed 42
