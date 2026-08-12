SHELL := /bin/bash
COMPOSE := podman compose -f docker/docker-compose.yml
PODMAN_SOCKET ?= /run/user/$(shell id -u)/podman/podman.sock
export PODMAN_SOCKET

.PHONY: doctor setup models package-lambda local-up local-down local-provision seed ingest-corpus smoke test test-unit test-property test-integration test-e2e lint types ci eval eval-langsmith release-check reset

doctor:
	@uv run --no-project --python 3.12 python scripts/doctor.py

setup:
	uv sync --frozen --all-extras --group dev
	cd infra/cdk && npm ci

models:
	uv run python scripts/download_models.py

package-lambda:
	./scripts/package_lambda.sh

local-up:
	mkdir -p .local/floci .models
	$(COMPOSE) up -d
	uv run python scripts/wait_local.py

local-down:
	$(COMPOSE) down

local-provision: package-lambda
	uv run python scripts/local/provision.py

seed:
	uv run python scripts/seed.py

ingest-corpus:
	uv run python scripts/ingest_corpus.py

smoke:
	uv run python scripts/smoke.py

lint:
	uv run ruff format --check .
	uv run ruff check .

types:
	uv run mypy src/

test-unit:
	uv run pytest tests/unit --cov=src/rag_ops_guard --cov-report=term-missing --cov-report=xml:artifacts/coverage.xml

test-property:
	uv run pytest tests/property

test-integration:
	uv run pytest -m integration tests/integration

test-e2e:
	uv run pytest -m e2e tests/e2e

test: test-unit test-property

ci: lint types test test-integration

eval:
	uv run --extra eval python evaluation/runners/run_golden.py
	uv run --extra eval python evaluation/runners/run_ragas.py

eval-langsmith:
	uv run --extra eval python evaluation/runners/run_langsmith.py

release-check: lint types test test-integration test-e2e eval
	cd infra/cdk && npm test && npx cdk synth

reset:
	-uv run python scripts/local/reset.py
	-$(COMPOSE) down -v
	rm -rf .local/floci .local/lambda-package .local/api-url artifacts/*
	touch artifacts/.gitkeep
