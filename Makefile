.PHONY: format lint test typecheck

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
