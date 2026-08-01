.PHONY: check format format-check install-dev lint test typecheck

PYTHON ?= python3

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff check --select I --fix .
	ruff format .

format-check:
	ruff format --check .

typecheck:
	ty check --exit-zero-on-warning

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

check: lint typecheck test
