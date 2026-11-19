.PHONY: install test lint format run ingest docker-build docker-run clean

PY ?= python3

install:            ## create a virtualenv with runtime + dev dependencies
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

test:               ## run the test-suite (offline; uses mock replicas of both sites)
	. .venv/bin/activate && pytest

lint:               ## static checks
	. .venv/bin/activate && ruff check app tests

format:
	. .venv/bin/activate && ruff check --fix app tests

ingest:             ## scrape both sources and build the search index into ./data
	. .venv/bin/activate && python -m app.cli ingest

run:                ## run the API + UI on http://localhost:7860
	. .venv/bin/activate && python -m app

docker-build:
	docker build -t dg-wasalt-assistant .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env dg-wasalt-assistant

clean:
	rm -rf data models .pytest_cache .ruff_cache
