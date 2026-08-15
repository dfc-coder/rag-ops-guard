#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVMS_HOST_PORT="${OVMS_HOST_PORT:-8083}"
CHAINLIT_VERSION="${CHAINLIT_VERSION:-2.11.1}"
OVMS_EMBEDDING_MODEL="${OVMS_EMBEDDING_MODEL:-OpenVINO/Qwen3-Embedding-0.6B-int8-ov}"
OVMS_RERANKER_MODEL="${OVMS_RERANKER_MODEL:-OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov}"
OPENVINO_VECTOR_INDEX="${OPENVINO_VECTOR_INDEX:-ops-knowledge-openvino-v1}"

printf '\n[1/5] Static contract + single-pipeline architecture\n'
uv run python -m py_compile \
  src/rag_ops_guard/agent/conversation.py \
  src/rag_ops_guard/app.py \
  src/rag_ops_guard/domain/models.py \
  src/rag_ops_guard/handlers/query.py \
  scripts/chainlit_react_ui.py \
  scripts/gradio_react_ui.py \
  scripts/calibrate_retrieval_admission.py \
  scripts/beta_freeze_smoke.py
uv run ruff format --check .
uv run ruff check .
uv run mypy src/
uv run pytest \
  tests/architecture/test_single_query_pipeline.py \
  tests/unit/test_query_response_contract.py \
  tests/unit/test_conversation_agent.py \
  tests/adversarial/test_deterministic_guards.py \
  -q
uv run python -c \
  'from rag_ops_guard.app import conversation_agent, query_workflow; assert conversation_agent() is query_workflow()'

printf '\n[2/5] Local runtime + corpus integrity\n'
make generation-model local-core-up openvino-up

export EMBEDDING_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export EMBEDDING_MODEL="$OVMS_EMBEDDING_MODEL"
export RERANKER_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export RERANKER_MODEL="$OVMS_RERANKER_MODEL"
export S3_VECTOR_INDEX="$OPENVINO_VECTOR_INDEX"
export RETRIEVAL_TOP_K="${BETA_RETRIEVAL_TOP_K:-8}"
export RETRIEVAL_CONTEXT_K="${BETA_RETRIEVAL_CONTEXT_K:-3}"
uv run python scripts/local/ensure_data.py

printf '\n[3/5] Calibrate post-retrieval evidence admission\n'
CALIBRATION_ENV="$(mktemp)"
uv run python scripts/calibrate_retrieval_admission.py --env-file "$CALIBRATION_ENV"
source "$CALIBRATION_ENV"
rm -f "$CALIBRATION_ENV"

printf '\n[4/5] Unified ReAct beta behavior\n'
uv run python scripts/beta_freeze_smoke.py

printf '\n[5/5] Client adapters import the same core\n'
uv run --with "chainlit==${CHAINLIT_VERSION}" python -c \
  'import runpy; runpy.run_path("scripts/chainlit_react_ui.py", run_name="beta_freeze_gate")'
uv run python -c 'import ast; ast.parse(open("scripts/gradio_react_ui.py", encoding="utf-8").read())'

cat <<'EOF'

BETA FREEZE GATE: PASS

Freeze invariants:
  - one canonical ConversationAgent owns conversation, memory, tool loop and streaming
  - Chainlit, Gradio and REST resolve that same application core
  - the generation model is actually bound to search_documents/list_documents
  - no semantic/scope router decides tool use in the canonical path
  - deterministic SafetyGuard runs before model/tool execution
  - post-retrieval relevance admission remains deterministic and calibrated
  - grounded answers require citations
  - ungrounded answers cannot carry citations
  - unsupported corpus-specific facts return insufficient_evidence
  - general knowledge/code stays ungrounded with zero document tools
  - Calypso grounded follow-up re-queries documents and preserves conversational context

This commit is eligible to freeze only after repository CI is green and the target Fedora/Tiger Lake
machine passes this gate plus a short manual Chainlit check.
EOF
