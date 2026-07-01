.PHONY: format lint test test-cov typecheck

format:
	isort .
	black .

lint:
	ruff check .
	black --check .
	isort --check-only .
	mypy video_analysis --ignore-missing-imports --no-strict-optional

typecheck:
	mypy video_analysis --ignore-missing-imports --no-strict-optional

test:
	pytest

# Used by CI (tests.yml) — produces coverage.xml for the artifact upload.
test-cov:
	pytest --cov=video_analysis --cov=backend/app --cov-report=term --cov-report=xml
