.PHONY: setup pipeline analysis export test lint all

setup:
	uv sync --locked

pipeline:
	uv run python src/pipeline.py

analysis:
	uv run python src/analysis.py

export:
	uv run python src/export_site.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

all: pipeline analysis export
