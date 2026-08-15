#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVMS_HOST_PORT="${OVMS_HOST_PORT:-8083}"
CHAINLIT_VERSION="${CHAINLIT_VERSION:-2.11.1}"
OVMS_EMBEDDING_MODEL="${OVMS_EMBEDDING_MODEL:-OpenVINO/Qwen3-Embedding-0.6B-int8-ov}"
OVMS_RERANKER_MODEL="${OVMS_RERANKER_MODEL:-OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov}"
OPENVINO_VECTOR_INDEX="${OPENVINO_VECTOR_INDEX:-ops-knowledge-openvino-v1}"

make generation-model local-core-up openvino-up

export EMBEDDING_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export EMBEDDING_MODEL="$OVMS_EMBEDDING_MODEL"
export RERANKER_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export RERANKER_MODEL="$OVMS_RERANKER_MODEL"
export S3_VECTOR_INDEX="$OPENVINO_VECTOR_INDEX"
export RETRIEVAL_TOP_K="${BETA_RETRIEVAL_TOP_K:-8}"
export RETRIEVAL_CONTEXT_K="${BETA_RETRIEVAL_CONTEXT_K:-3}"

printf '\n[1/5] Local corpus integrity\n'
uv run python scripts/local/ensure_data.py

printf '\n[2/5] Streaming / rollback / grounding policy regressions\n'
uv run pytest \
  tests/unit/test_react_streaming.py \
  tests/unit/test_grounding_policy.py \
  tests/unit/test_resilient_retrieval.py \
  -q

printf '\n[3/5] Direct chat/code streaming\n'
uv run python scripts/react_direct_ui_smoke.py

printf '\n[4/5] Conversational Grounding v2: Calypso + follow-up + evidence reuse\n'
uv run python scripts/react_rag_smoke.py

printf '\n[5/5] Chainlit application import/config\n'
uv run --with "chainlit==${CHAINLIT_VERSION}" python -c \
  'import runpy; runpy.run_path("scripts/chainlit_react_ui.py", run_name="chainlit_migration_gate")'

cat <<'EOF'

CHAINLIT MIGRATION GATE: HEADLESS PASS

Manual UI confirmation still required before merge:
  1. make chainlit-beta
  2. open http://127.0.0.1:8001
  3. direct code streams with policy=direct and no RAG
  4. Calypso retries answers 3 and exposes source cards
  5. follow-up after the third re-grounds and answers Treasury Integrations
  6. "Resumilo" reuses the active evidence window without another RAG call
  7. Stop interrupts a generation without corrupting prior message/grounding state
  8. Retry works after a recoverable failure
  9. environment selector and Local Status behave correctly

If those pass, this branch is ready to merge into the Chainlit migration candidate.
EOF
