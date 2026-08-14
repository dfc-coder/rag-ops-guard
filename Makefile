SHELL := /bin/bash
COMPOSE := podman compose -f docker/docker-compose.yml
PODMAN_SOCKET ?= /run/user/$(shell id -u)/podman/podman.sock
CACHE_HOME ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)
MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/models
UI_GRADIO_VERSION ?= 6.20.0
BENCH_REQUESTS ?= 5
BENCH_CONCURRENCY ?= 1
RETRIEVAL_TOP_K ?= 20
RETRIEVAL_CONTEXT_K ?= 4
ROUTER_MIN_SCORE ?= 0.35
ROUTER_MIN_MARGIN ?= 0.015
LLM_ANSWER_MAX_TOKENS ?= 512
LLM_TEMPERATURE ?= 0.7
LLM_TOP_P ?= 0.8
LLM_TOP_K ?= 20
LLM_MIN_P ?= 0.0
LLM_PRESENCE_PENALTY ?= 1.5
LLM_REPEAT_PENALTY ?= 1.0
export PODMAN_SOCKET MODEL_DIR LLAMA_CTX_SIZE LLAMA_PARALLEL RETRIEVAL_TOP_K RETRIEVAL_CONTEXT_K ROUTER_MIN_SCORE ROUTER_MIN_MARGIN LLM_ANSWER_MAX_TOKENS LLM_TEMPERATURE LLM_TOP_P LLM_TOP_K LLM_MIN_P LLM_PRESENCE_PENALTY LLM_REPEAT_PENALTY

.PHONY: doctor setup models package-lambda local-up local-down local-data retrieval-calibrate local-provision seed ingest-corpus smoke demo demo-prepare demo-query ui ui-init demo-ready demo-client benchmark benchmark-api test test-unit test-property test-integration test-e2e lint types ci eval eval-langsmith release-check reset

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
	mkdir -p .local/floci "$(MODEL_DIR)"
	$(COMPOSE) up -d
	uv run python scripts/wait_local.py

local-down:
	$(COMPOSE) down

# Fail-closed local corpus check. Rebuilds when any repository document manifest
# is stale/missing or any manifest-referenced vector is absent from Floci.
local-data:
	uv run python scripts/local/ensure_data.py

# Data-driven admission calibration. Reuses a cached artifact only when the
# reranker model, labeled dataset, and knowledge-base fingerprint all match.
retrieval-calibrate:
	uv run python scripts/calibrate_retrieval.py

local-provision: package-lambda
	uv run python scripts/local/provision.py

seed:
	uv run python scripts/seed.py

ingest-corpus:
	uv run python scripts/ingest_corpus.py

smoke:
	uv run python scripts/smoke.py

demo-prepare: local-provision seed
	uv run python scripts/demo_prepare.py

demo: models local-up demo-prepare
	uv run python scripts/demo.py "What is the current maximum automated retry count for Calypso timeouts?" --system payments --environment production

# Usage: make demo-query QUESTION='Can I retry a Calypso payment?' SYSTEM=payments ENVIRONMENT=production
demo-query:
	@test -n "$(QUESTION)" || { echo 'QUESTION is required'; exit 2; }
	uv run python scripts/demo.py "$(QUESTION)" $(if $(SYSTEM),--system "$(SYSTEM)",) $(if $(ENVIRONMENT),--environment "$(ENVIRONMENT)",)

# Developer UI. It still requires a valid retrieval calibration so developers
# exercise the same admission contract used by the client demo.
ui: models local-up local-data retrieval-calibrate
	uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

ui-init: models local-up local-data retrieval-calibrate
	uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Client gate: real Floci + embeddings + calibrated multilingual reranker + Qwen + multi-turn assertions.
demo-ready: models local-up local-data retrieval-calibrate
	uv run python scripts/demo_ready.py

# Client-facing entrypoint. Gradio is not launched if demo-ready fails.
demo-client: demo-ready
	uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Benchmarks the same in-process workflow used by Gradio. It is safe to run
# standalone: containers, local vector data, and retrieval calibration are ensured first.
benchmark: models local-up local-data retrieval-calibrate
	uv run python scripts/benchmark_runtime.py --transport direct --requests $(BENCH_REQUESTS) --concurrency $(BENCH_CONCURRENCY)

# Optional full local API benchmark. Provisioning is required only for this path.
benchmark-api: models local-up local-data retrieval-calibrate local-provision
	uv run python scripts/benchmark_runtime.py --transport api --requests $(BENCH_REQUESTS) --concurrency $(BENCH_CONCURRENCY)

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
	rm -rf .local/floci .local/lambda-package .local/api-url .local/reranker-calibration.json artifacts/*
	touch artifacts/.gitkeep
