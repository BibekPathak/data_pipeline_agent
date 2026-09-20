.PHONY: install dev install-dev test unit integration agent safety evaluation evals run api clean lint typecheck

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

unit:
	pytest tests/unit

integration:
	pytest tests/integration

agent:
	pytest tests/agent

safety:
	pytest tests/safety

evaluation:
	pytest tests/evaluation

# Full evaluation against all benchmark + adversarial scenarios
evals:
	python -m app.evaluation.runner --scenarios all --report reports/evaluation.md

run:
	uvicorn app.api.app:app --reload

api:
	uvicorn app.api.app:app --host 0.0.0.0 --port 8000

lint:
	ruff check app tests

typecheck:
	mypy app

clean:
	rm -rf .pytest_cache .coverage htmlcov reports .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
