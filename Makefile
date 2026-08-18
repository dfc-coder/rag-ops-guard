SHELL := /bin/bash
UV_PYTHON ?= 3.12
COMPOSE := podman compose -f docker/docker-compose.yml
PODMAN_API_SOCKET ?= /run/user/$(shell id -u)/rag-ops-guard/podman-api.sock
CACHE_HOME ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)
MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/models
OVMS_MODEL_DIR ?= $(CACHE_HOME)/rag-ops-guard/openvino-models
COMPOSE_PROJECT_NAME ?= rag-ops-guard
RAG_OPS_NETWORK ?= rag-ops-net
FLOCI_IMAGE ?= localhost/rag-ops-floci-jvm:1.6.0
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
CHAINLIT_HOST_PORT ?= 8001
CHAINLIT_POC_PORT ?= 8001
BENCH_REQUESTS ?= 5
BENCH_CONCURRENCY ?= 1
RETRIEVAL_TOP_K ?= 20
RETRIEVAL_CONTEXT_K ?= 4
BETA_RETRIEVAL_TOP_K ?= 8
BETA_RETRIEVAL_CONTEXT_K ?= 3
RETRIEVAL_DOMAIN_MIN_RELEVANCE ?= 0.50
RETRIEVAL_MIN_RELEVANCE ?= 0.50
LLM_ANSWER_MAX_TOKENS ?= 512
LLM_TEMPERATURE ?= 0.7
LLM_TOP_P ?= 0.8
LLM_TOP_K ?= 20
LLM_MIN_P ?= 0.0
LLM_PRESENCE_PENALTY ?= 1.5
LLM_REPEAT_PENALTY ?= 1.0
EMBEDDING_TIMEOUT_SECONDS ?= 60
RERANKER_TIMEOUT_SECONDS ?= 90
GOLDEN_HTTP_TIMEOUT_SECONDS ?= 180
GOLDEN_CASE_TIMEOUT_SECONDS ?= 240
RAGAS_JUDGE_TIMEOUT_SECONDS ?= 120
RAGAS_OPERATION_TIMEOUT_SECONDS ?= 120
RAGAS_MAX_RETRIES ?= 2
RAGAS_MAX_WAIT_SECONDS ?= 5
RAGAS_MAX_WORKERS ?= 2
RAGAS_SUITE_TIMEOUT_SECONDS ?= 1800
LAMBDA_PYTHON_VERSION ?= 3.12
export UV_PYTHON PODMAN_API_SOCKET MODEL_DIR OVMS_MODEL_DIR COMPOSE_PROJECT_NAME RAG_OPS_NETWORK FLOCI_IMAGE FLOCI_CONTAINER_NAME LLAMA_GEN_CONTAINER_NAME LLAMA_EMBED_CONTAINER_NAME LLAMA_RERANK_CONTAINER_NAME OVMS_CONTAINER_NAME FLOCI_HOST_PORT LLAMA_GEN_HOST_PORT LLAMA_EMBED_HOST_PORT LLAMA_RERANK_HOST_PORT OVMS_HOST_PORT OVMS_IMAGE OVMS_EMBEDDING_MODEL OVMS_RERANKER_MODEL LLAMA_CTX_SIZE LLAMA_PARALLEL RETRIEVAL_TOP_K RETRIEVAL_CONTEXT_K RETRIEVAL_DOMAIN_MIN_RELEVANCE RETRIEVAL_MIN_RELEVANCE LLM_ANSWER_MAX_TOKENS LLM_TEMPERATURE LLM_TOP_P LLM_TOP_K LLM_MIN_P LLM_PRESENCE_PENALTY LLM_REPEAT_PENALTY EMBEDDING_TIMEOUT_SECONDS RERANKER_TIMEOUT_SECONDS GOLDEN_HTTP_TIMEOUT_SECONDS GOLDEN_CASE_TIMEOUT_SECONDS RAGAS_JUDGE_TIMEOUT_SECONDS RAGAS_OPERATION_TIMEOUT_SECONDS RAGAS_MAX_RETRIES RAGAS_MAX_WAIT_SECONDS RAGAS_MAX_WORKERS RAGAS_SUITE_TIMEOUT_SECONDS LAMBDA_PYTHON_VERSION

OPENVINO_BACKEND_ENV := EMBEDDING_BASE_URL=http://127.0.0.1:$(OVMS_HOST_PORT)/v3 EMBEDDING_MODEL=$(OVMS_EMBEDDING_MODEL) EMBEDDING_TIMEOUT_SECONDS=$(EMBEDDING_TIMEOUT_SECONDS) RERANKER_BASE_URL=http://127.0.0.1:$(OVMS_HOST_PORT)/v3 RERANKER_MODEL=$(OVMS_RERANKER_MODEL) RERANKER_TIMEOUT_SECONDS=$(RERANKER_TIMEOUT_SECONDS) S3_VECTOR_INDEX=$(OPENVINO_VECTOR_INDEX) RETRIEVAL_TOP_K=$(BETA_RETRIEVAL_TOP_K) RETRIEVAL_CONTEXT_K=$(BETA_RETRIEVAL_CONTEXT_K)
OPENVINO_ENV := $(OPENVINO_BACKEND_ENV) RETRIEVAL_DOMAIN_MIN_RELEVANCE=$(RETRIEVAL_DOMAIN_MIN_RELEVANCE) RETRIEVAL_MIN_RELEVANCE=$(RETRIEVAL_MIN_RELEVANCE)
LAMBDA_OPENVINO_ENV := LAMBDA_EMBEDDING_BASE_URL=http://$(OVMS_CONTAINER_NAME):8000/v3 LAMBDA_EMBEDDING_MODEL=$(OVMS_EMBEDDING_MODEL) LAMBDA_RERANKER_BASE_URL=http://$(OVMS_CONTAINER_NAME):8000/v3 LAMBDA_RERANKER_MODEL=$(OVMS_RERANKER_MODEL)

.PHONY: up down help status connectivity config-bootstrap config-publish config-shadow-check golden golden-all logs doctor setup models generation-model package-lambda local-up local-core-up local-down local-clean local-data retrieval-validate local-provision seed ingest-corpus smoke demo demo-prepare demo-query ui ui-init beta openvino-models openvino-up openvino-down openvino-status openvino-smoke beta-openvino beta-react gradio-react chainlit-beta chainlit-gate physical-up physical-generation-contract physical-relevance-calibrate physical-ready physical-eval physical-eval-measure physical-smoke demo-ready demo-client benchmark benchmark-api test test-unit test-property test-integration test-e2e lint lint-advisory types ci eval eval-measure eval-human-review eval-judge-calibrate eval-langsmith release-check reset

help:
	@echo 'Canonical local workflow:'
	@echo '  make up                         # start/provision the complete physical stack'
	@echo '  make ui                         # interactive UI through Floci -> Lambda -> Agent'
	@echo '  make config-publish REASON="..." # publish append-only config revision'
	@echo '  make config-shadow-check        # compare DB revision with publisher environment'
	@echo '  make connectivity               # execute dependency probes from Lambda and API Gateway'
	@echo '  make golden CASE=<id>           # one Golden through the same /v1/query'
	@echo '  make golden-all                 # all 34 Golden cases through the same /v1/query'
	@echo '  make status                     # runtime/model/API/config-hash state'
	@echo '  make logs                       # recent runtime logs'
	@echo '  make down                       # stop the local runtime'

status:
	@uv run python scripts/local/ops.py status

connectivity:
	@uv run python scripts/local/connectivity.py

config-bootstrap:
	@set -a; [ ! -f .env ] || source .env; set +a; uv run python -c 'from scripts.local.provision import ensure_s3, ensure_config_table; ensure_s3(); ensure_config_table()'
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=env $(OPENVINO_BACKEND_ENV) uv run python scripts/config_publish.py --reason "physical runtime baseline" --actor "$${USER:-local}"

config-publish:
	@test -n "$(REASON)" || { echo 'REASON is required'; exit 2; }
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=env uv run python scripts/config_publish.py --reason "$(REASON)" --actor "$${USER:-local}"

config-shadow-check:
	@set -a; [ ! -f .env ] || source .env; set +a; uv run python scripts/config_shadow_check.py

up:
	@test -f .env || { echo 'Missing .env. Copy/configure .env before starting the runtime.'; exit 2; }
	@set -a; source .env; set +a; $(MAKE) physical-ready
	@$(MAKE) status

down: local-down

golden:
	@$(MAKE) connectivity
	@uv run --extra eval python scripts/local/ops.py golden $(if $(CASE),--case "$(CASE)",) $(if $(FORCE),--force,)

golden-all:
	@$(MAKE) connectivity
	@uv run --extra eval python scripts/local/ops.py golden --all $(if $(FORCE),--force,)

logs:
	@uv run python scripts/local/ops.py logs

doctor:
	@uv run --no-project --python 3.12 python scripts/doctor.py

setup:
	uv sync --frozen --all-extras --group dev
	cd infra/cdk && npm ci

models:
	uv run python scripts/download_models.py

generation-model:
	MODEL_FILES=Qwen3.5-2B-UD-Q4_K_XL.gguf uv run python scripts/download_models.py

package-lambda:
	./scripts/package_lambda.sh

local-up:
	mkdir -p "$(MODEL_DIR)"
	uv run python scripts/local/floci_jvm_image.py ensure --image "$(FLOCI_IMAGE)"
	uv run python scripts/local/podman_api_service.py start --socket "$(PODMAN_API_SOCKET)"
	$(COMPOSE) up -d
	uv run python scripts/wait_local.py

local-core-up:
	mkdir -p "$(MODEL_DIR)"
	uv run python scripts/local/floci_jvm_image.py ensure --image "$(FLOCI_IMAGE)"
	uv run python scripts/local/podman_api_service.py start --socket "$(PODMAN_API_SOCKET)"
	-$(COMPOSE) stop llama-embed llama-rerank
	$(COMPOSE) up -d floci llama-gen
	uv run python scripts/wait_core.py

local-down:
	-uv run python scripts/openvino_runtime.py down
	-$(COMPOSE) down
	-uv run python scripts/local/podman_api_service.py stop --socket "$(PODMAN_API_SOCKET)"

local-clean:
	-uv run python scripts/openvino_runtime.py down
	-$(COMPOSE) down -v
	-uv run python scripts/local/podman_api_service.py stop --socket "$(PODMAN_API_SOCKET)"

local-data:
	uv run python scripts/local/ensure_data.py

retrieval-validate:
	uv run python scripts/validate_retrieval.py

local-provision: package-lambda
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) $(LAMBDA_OPENVINO_ENV) uv run python scripts/local/provision.py
	@uv run python scripts/local/connectivity.py

seed:
	uv run python scripts/seed.py

ingest-corpus:
	uv run python scripts/ingest_corpus.py

smoke:
	uv run python scripts/smoke.py

demo-prepare: local-provision seed
	uv run python scripts/demo_prepare.py

demo: up
	uv run python scripts/demo.py "What is the current maximum automated retry count for Calypso timeouts?" --system payments --environment production

demo-query:
	@test -n "$(QUESTION)" || { echo 'QUESTION is required'; exit 2; }
	@test -s .local/api-url || { echo 'Canonical API is not provisioned. Run: make up'; exit 2; }
	uv run python scripts/demo.py "$(QUESTION)" $(if $(SYSTEM),--system "$(SYSTEM)",) $(if $(ENVIRONMENT),--environment "$(ENVIRONMENT)",)

ui:
	@test -s .local/api-url || { echo 'Canonical API is not provisioned. Run: make up'; exit 2; }
	@CHAINLIT_HOST_PORT=$(CHAINLIT_HOST_PORT) CHAINLIT_VERSION=$(CHAINLIT_VERSION) bash scripts/run_chainlit_beta.sh

ui-init: up
	@$(MAKE) ui

beta: models local-up local-data
	RETRIEVAL_TOP_K=$(BETA_RETRIEVAL_TOP_K) RETRIEVAL_CONTEXT_K=$(BETA_RETRIEVAL_CONTEXT_K) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

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

beta-openvino: generation-model local-core-up openvino-up
	CONFIG_SOURCE=env $(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	CONFIG_SOURCE=env $(OPENVINO_ENV) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_ui.py

beta-react: ui

gradio-react: generation-model local-core-up openvino-up
	CONFIG_SOURCE=env $(OPENVINO_ENV) uv run python scripts/local/ensure_data.py
	CONFIG_SOURCE=env $(OPENVINO_ENV) uv run --with "gradio==$(UI_GRADIO_VERSION)" python scripts/gradio_react_ui.py

chainlit-beta: ui

chainlit-gate:
	CHAINLIT_VERSION=$(CHAINLIT_VERSION) bash scripts/chainlit_migration_gate.sh

physical-up: generation-model local-core-up openvino-models
	OVMS_NETWORK=$(RAG_OPS_NETWORK) uv run python scripts/openvino_runtime.py up
	@$(MAKE) config-bootstrap
	CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run python scripts/local/ensure_data.py

physical-generation-contract: physical-up
	CONFIG_SOURCE=env uv run python scripts/validate_llama_contract.py

physical-relevance-calibrate: physical-generation-contract
	$(OPENVINO_BACKEND_ENV) uv run python scripts/calibrate_relevance_floors.py --env-file .local/relevance-floors.env --publish --reason "physical relevance calibration"
	@cat .local/relevance-floors.env

physical-ready: physical-relevance-calibrate package-lambda
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run python scripts/validate_retrieval.py
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) $(LAMBDA_OPENVINO_ENV) uv run python scripts/local/provision.py
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run python scripts/ingest_corpus.py
	@uv run python scripts/local/connectivity.py

physical-eval-measure: physical-ready
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run --extra eval python evaluation/runners/run_golden.py
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) RAGAS_REQUIRE_CALIBRATION=0 uv run --extra eval python evaluation/runners/run_ragas.py
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run --extra eval python scripts/prepare_judge_human_review.py

physical-eval: physical-ready
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run --extra eval python evaluation/runners/run_golden.py
	@set -a; [ ! -f .env ] || source .env; set +a; CONFIG_SOURCE=db $(OPENVINO_BACKEND_ENV) uv run --extra eval python evaluation/runners/run_ragas.py

physical-smoke: physical-ready
	CONFIG_SOURCE=db uv run python scripts/smoke.py

demo-ready: models local-up local-data retrieval-validate
	uv run python scripts/demo_ready.py

demo-client: ui

benchmark: models local-up local-data retrieval-validate
	uv run python scripts/benchmark_runtime.py --transport direct --requests $(BENCH_REQUESTS) --concurrency $(BENCH_CONCURRENCY)

benchmark-api: models local-up local-data retrieval-validate local-provision
	uv run python scripts/benchmark_runtime.py --transport api --requests $(BENCH_REQUESTS) --concurrency $(BENCH_CONCURRENCY)

lint:
	uv run ruff format --check .
	uv run ruff check .

lint-advisory:
	-uv run ruff format --check .
	-uv run ruff check .

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

ci: lint-advisory types test test-integration

eval:
	uv run --extra eval python evaluation/runners/run_golden.py
	uv run --extra eval python evaluation/runners/run_ragas.py

eval-measure:
	uv run --extra eval python evaluation/runners/run_golden.py
	RAGAS_REQUIRE_CALIBRATION=0 uv run --extra eval python evaluation/runners/run_ragas.py

eval-human-review:
	uv run --extra eval python scripts/prepare_judge_human_review.py

eval-judge-calibrate:
	uv run --extra eval python scripts/calibrate_ragas_judge.py

eval-langsmith:
	uv run --extra eval python evaluation/runners/run_langsmith.py

release-check: lint-advisory types test test-integration test-e2e eval
	cd infra/cdk && npm test && npx cdk synth

reset:
	-uv run python scripts/openvino_runtime.py down
	-uv run python scripts/local/reset.py
	-$(COMPOSE) down -v
	-uv run python scripts/local/podman_api_service.py stop --socket "$(PODMAN_API_SOCKET)"
	rm -rf .local/lambda-build .local/lambda-package .local/lambda-package.zip .local/api-url artifacts/*
	touch artifacts/.gitkeep