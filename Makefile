.PHONY: setup pipeline analysis test lint all

setup:
	python -m pip install -r requirements.txt

pipeline:
	python src/pipeline.py

analysis:
	python src/analysis.py

test:
	python -m pytest -q

lint:
	python -m ruff check .

all: pipeline analysis
