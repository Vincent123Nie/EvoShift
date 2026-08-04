.PHONY: install test lint format typecheck build demo doctor clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=evoshift --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src/evoshift

build:
	python -m build

doctor:
	python -m evoshift doctor

demo:
	python -m evoshift run --config configs/experiments/offline_demo.yaml

clean:
	python scripts/clean_artifacts.py

