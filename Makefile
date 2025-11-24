.PHONY: format lint test

format:
	isort video_analysis examples tests
	black video_analysis examples tests

lint:
	ruff check video_analysis examples tests

test:
	pytest
