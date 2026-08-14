SHELL := /bin/bash
COMPOSE := podman compose -f docker/docker-compose.yml
PODMAN_SOCKET ?= /run/user/$(shell id -u)/podman/podman.sock
CACHE_HOME ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)
MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/models
COMPOSE_PROJECT_NAME ?= rag-ops-guard
RAG_OPS_NETWORK ?= rag-ops-net
FLOCI_CONTAINER_NAME ?= rag-ops-floci
LLAMA_GEN_CONTAINER_NAME ?= rag-ops-llama-gen
LLAMA_EMBED_CONTAINER_NAME ?= rag-ops-llama-embed
LLAMA_RERANK_CONTAINER_NAME ?= rag-ops-llama-rerank
FLOCI_HOST_PORT ?= 4566
LLAMA_GEN_HOST_PORT ?= 8080
LLAMA_EMBED_HOST_PORT ?= 8081
LLAMA_RERANK_HOST_PORT ?= 8082
UI_GRADIO_VERSION ?= 6.20.0
BENCH_REQUESTS ?= 5
BENCH_CONCURRENCY ?= 1
RETRIEVAL_TOP_K ?= 20
RETRIEVAL_CONTEXT_K ?= 4
BETA_RETRIEVAL_TOP_K ?= 8
BETA_RETRIEVAL_CONTEXT_K ?= 3
ROUTER_MIN_SCORE ?= 0.35
ROUTER_MIN_MARGIN ?= 0.015
LLM_ANSWER_MAX_TOKENS ?= 512
LLM_TEMPERATURE ?= 0.7
LLM_TOP_P ?= 0.8
LLM_TOP_K ?= 20
LLM_MIN_P ?= 0.0
LLM_PRESENCE_PENALTY ?= 1.5
LLM_REPEAT_PENALTY ?= 1.0
export PODMAN_SOCKET MODEL_DIR COMPOSE_PROJECT_NAME RAG_OPS_NETWORK FLOCI_CONTAINER_NAME LLAMA_GEN_CONTAINER_NAME LLAMA_EMBED_CONTAINER_NAME LLAMA_RERANK_CONTAINER_NAME FLOCI_HOST_PORT LLAMA_GEN_HOST_PORT LLAMA_EMBED_HOST_PORT LLAMA_RERANK_HOST_PORT LLAMA_CTX_SIZE LLAMA_PARALLEL RETRIEVAL_TOP_K RETRIEVAL_CONTEXT_K ROUTER_MIN_SCORE ROUTER_MIN_MARGIN LLM_ANSWER_MAX_TOKENS LLM_TEMPERATURE LLM_TOP_P LLM_TOP_K LLM_MIN_P LLM_PRESENCE_PENALTY LLM_REPEAT_PENALTY

.PHONY: doctor setup models package-lambda local-up local-down local-clean local-data retrieval-validate local-provision seed ingest-corpus smoke demo demo-prepare demo-query ui ui-init beta demo-ready demo-client benchmark benchmark-api test test-unit test-property test-integration test-e2e lint types ci eval eval-langsmith release-check reset

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
	mkdir -p "$(MODEL_DIR)"
	$(COMPOSE) up -d
	uv run python scripts/wait_local.py

local-down:
	$(COMPOSE) down

local-clean:
	$(COMPOSE) down -v

# Fail-closed local corpus check. Rebuilds when any repository document manifest
# is stale/missing or any manifest-referenced vector is absent from Floci.
local-data:
	uv run python scripts/local/ensure_data.py

# Behavioral retrieval validation. The learned relevance grader must admit expected
# documents for positives and reject all documents for hard negatives.
retrieval-validate:
	uv run python scripts/validate_retrieval.py

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

# Developer UI keeps the full retrieval validation guard.
ui: models local-up local-data retrieval-validate
	uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

ui-init: models local-up local-data retrieval-validate
	uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Fast beta path: boot the real local runtime and corpus, then open Gradio.
# Keep the learned reranker, but rerank only the strongest 8 candidates instead of 20.
beta: models local-up local-data
	RETRIEVAL_TOP_K=$(BETA_RETRIEVAL_TOP_K) RETRIEVAL_CONTEXT_K=$(BETA_RETRIEVAL_CONTEXT_K) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Client gate: real Floci + embeddings + Qwen relevance grader + Qwen 2B + multi-turn assertions.
demo-ready: models local-up local-data retrieval-validate
	uv run python scripts/demo_ready.py

# Client-facing beta entrypoint. Full validation remains available through demo-ready.
demo-client: beta

# Benchmarks the same in-process workflow used by Gradio. It is safe to run
# standalone: containers, local vector data, and retrieval validation are ensured first.
benchmark: models local-up local-data retrieval-validate
	uv run python scripts/benchmark_runtime.py --transport direct --requests $(BENCH_REQUESTS) --concurrency $(BENCH_CONCURRENCY)

# Optional full local API benchmark. Provisioning is required only for this path.
benchmark-api: models local-up local-data retrieval-validate local-provision
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
	rm -rf .local/lambda-package .local/api-url artifacts/*
	touch artifacts/.gitkeep
