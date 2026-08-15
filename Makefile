SHELL := /bin/bash
COMPOSE := podman compose -f docker/docker-compose.yml
PODMAN_SOCKET ?= /run/user/$(shell id -u)/podman/podman.sock
CACHE_HOME ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)
MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/models
OVMS_MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/openvino-models
COMPOSE_PROJECT_NAME ?= rag-ops-guard
RAG_OPS_NETWORK ?= rag-ops-net
FLOCI_CONTAINER_NAME ?= rag-ops-floci
LLAMA_GEN_CONTAINER_NAME ?= rag-ops-llama-gen
LLAMA_EMBED_CONTAINER_NAME ?= rag-ops-llama-embed
LLAMA_RERANK_CONTAINER_NAME ?= rag-ops-llama-rerank
OVMS_CONTAINER_NAME ?= rag-ops-ovms-rag
FLOCI_HOST_PORT ?= 4566
LLAMA_GEN_HOST_PORT ?= 8080
LLAMA_EMBED_HOST_PORT ?= 8081
LLAMA_RERANK_HOST_PORT ?= 8082
OVMS_HOST_PORT ?= 8083
OVMS_IMAGE ?= docker.io/openvino/model_server:latest-gpu
OVMS_EMBEDDING_MODEL ?= OpenVINO/Qwen3-Embedding-0.6B-int8-ov
OVMS_RERANKER_MODEL ?= OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov
OPENVINO_VECTOR_INDEX ?= ops-knowledge-openvino-v1
UI_GRADIO_VERSION ?= 6.20.0
CHAINLIT_VERSION ?= 2.11.1
CHAINLIT_HOST_PORT ?= 8000
CHAINLIT_POC_PORT ?= 8001
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
export PODMAN_SOCKET MODEL_DIR OVMS_MODEL_DIR COMPOSE_PROJECT_NAME RAG_OPS_NETWORK FLOCI_CONTAINER_NAME LLAMA_GEN_CONTAINER_NAME LLAMA_EMBED_CONTAINER_NAME LLAMA_RERANK_CONTAINER_NAME OVMS_CONTAINER_NAME FLOCI_HOST_PORT LLAMA_GEN_HOST_PORT LLAMA_EMBED_HOST_PORT LLAMA_RERANK_HOST_PORT OVMS_HOST_PORT OVMS_IMAGE OVMS_EMBEDDING_MODEL OVMS_RERANKER_MODEL LLAMA_CTX_SIZE LLAMA_PARALLEL RETRIEVAL_TOP_K RETRIEVAL_CONTEXT_K ROUTER_MIN_SCORE ROUTER_MIN_MARGIN LLM_ANSWER_MAX_TOKENS LLM_TEMPERATURE LLM_TOP_P LLM_TOP_K LLM_MIN_P LLM_PRESENCE_PENALTY LLM_REPEAT_PENALTY

OPENVINO_ENV := EMBEDDING_BASE_URL=http://127.0.0.1:$(OVMS_HOST_PORT)/v3 EMBEDDING_MODEL=$(OVMS_EMBEDDING_MODEL) RERANKER_BASE_URL=http://127.0.0.1:$(OVMS_HOST_PORT)/v3 RERANKER_MODEL=$(OVMS_RERANKER_MODEL) S3_VECTOR_INDEX=$(OPENVINO_VECTOR_INDEX) RETRIEVAL_TOP_K=$(BETA_RETRIEVAL_TOP_K) RETRIEVAL_CONTEXT_K=$(BETA_RETRIEVAL_CONTEXT_K)

.PHONY: doctor setup models generation-model package-lambda local-up local-core-up local-down local-clean local-data retrieval-validate local-provision seed ingest-corpus smoke demo demo-prepare demo-query ui ui-init beta openvino-models openvino-up openvino-down openvino-status openvino-smoke react-smoke react-rag-smoke react-direct-stream-smoke react-direct-ui-smoke beta-openvino beta-react gradio-react chainlit-beta chainlit-gate demo-ready demo-client benchmark benchmark-api test test-unit test-property test-integration test-e2e lint types ci eval eval-langsmith release-check reset

doctor:
	@uv run --no-project --python 3.12 python scripts/doctor.py

setup:
	uv sync --frozen --all-extras --group dev
	cd infra/cdk && npm ci

models:
	uv run python scripts/download_models.py

generation-model:
	MODEL_FILES=Qwen3.5-2B-Q4_K_M.gguf uv run python scripts/download_models.py

package-lambda:
	./scripts/package_lambda.sh

local-up:
	mkdir -p "$(MODEL_DIR)"
	$(COMPOSE) up -d
	uv run python scripts/wait_local.py

# CPU owns Python/LangGraph orchestration, Floci, BM25/RRF and Qwen generation.
# The legacy llama.cpp embedding/reranking services are stopped because OpenVINO owns those
# workloads in the hardware-split beta.
local-core-up:
	mkdir -p "$(MODEL_DIR)"
	-$(COMPOSE) stop llama-embed llama-rerank
	$(COMPOSE) up -d floci llama-gen
	uv run python scripts/wait_core.py

local-down:
	-uv run python scripts/openvino_runtime.py down
	$(COMPOSE) down

local-clean:
	-uv run python scripts/openvino_runtime.py down
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

# Fast legacy CPU-only beta path.
beta: models local-up local-data
	RETRIEVAL_TOP_K=$(BETA_RETRIEVAL_TOP_K) RETRIEVAL_CONTEXT_K=$(BETA_RETRIEVAL_CONTEXT_K) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Prepare the OpenVINO embedding/reranking model repository once. Models are cached under
# ~/.cache/rag-ops-guard/openvino-models and configured for the Intel GPU by OVMS.
openvino-models:
	uv run python scripts/openvino_prepare.py

openvino-up: openvino-models
	uv run python scripts/openvino_runtime.py up

openvino-down:
	uv run python scripts/openvino_runtime.py down

openvino-status:
	uv run python scripts/openvino_runtime.py status

openvino-smoke:
	uv run python scripts/openvino_smoke.py

# Legacy headless acceptance test for conversation + RAG tool + follow-up memory.
react-smoke: generation-model local-core-up openvino-up
	$(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	$(OPENVINO_ENV) uv run python scripts/react_smoke.py

# Current RAG acceptance test: exact Calypso regression + streaming + follow-up memory.
react-rag-smoke: generation-model local-core-up openvino-up
	$(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	$(OPENVINO_ENV) uv run python scripts/react_rag_smoke.py

# Regression probe for the original long direct-code path.
react-direct-stream-smoke: generation-model local-core-up
	uv run python scripts/react_direct_stream_smoke.py

# Short direct-code migration probe: requires immediate status, visible streaming and zero tools.
react-direct-ui-smoke: generation-model local-core-up
	uv run python scripts/react_direct_ui_smoke.py

# Hardware-split legacy UI: CPU generation/Floci/Python; Intel iGPU embeddings/reranking.
beta-openvino: generation-model local-core-up openvino-up
	$(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	$(OPENVINO_ENV) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

# Primary client beta UI: Chainlit + conversational LangGraph ReAct.
# CPU: Python/LangGraph + BM25/RRF + Floci + Qwen 3.5 2B generation.
# Intel Iris Xe/OpenVINO: Qwen3 embeddings + Qwen3 reranker.
beta-react: generation-model local-core-up openvino-up
	$(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	$(OPENVINO_ENV) uv run --with "chainlit==$(CHAINLIT_VERSION)" chainlit run scripts/chainlit_react_ui.py --host 127.0.0.1 --port $(CHAINLIT_HOST_PORT) --headless

# Gradio stays available as a developer/debug fallback during the Chainlit migration window.
gradio-react: generation-model local-core-up openvino-up
	$(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	$(OPENVINO_ENV) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_react_ui.py

# Side-by-side Chainlit candidate on 8001 without changing the normal beta port.
chainlit-beta:
	CHAINLIT_HOST_PORT=$(CHAINLIT_POC_PORT) CHAINLIT_VERSION=$(CHAINLIT_VERSION) bash scripts/run_chainlit_beta.sh

# Mandatory headless migration gate before promoting this branch to develop.
chainlit-gate:
	CHAINLIT_VERSION=$(CHAINLIT_VERSION) bash scripts/chainlit_migration_gate.sh

# Client gate: real Floci + embeddings + Qwen relevance grader + Qwen 2B + multi-turn assertions.
demo-ready: models local-up local-data retrieval-validate
	uv run python scripts/demo_ready.py

# Client-facing legacy beta entrypoint. Full validation remains available through demo-ready.
demo-client: beta

# Benchmarks the same in-process workflow used by the application.
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
	-uv run python scripts/openvino_runtime.py down
	-uv run python scripts/local/reset.py
	-$(COMPOSE) down -v
	rm -rf .local/lambda-package .local/api-url artifacts/*
	touch artifacts/.gitkeep
