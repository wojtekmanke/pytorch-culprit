PY := .venv/bin/python

.PHONY: check test lint types format report quickstart

check: lint types test

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

types:
	$(PY) -m mypy

test:
	$(PY) -m pytest -q

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

report:
	$(PY) -m culprit.report

quickstart:
	$(PY) examples/quickstart.py
