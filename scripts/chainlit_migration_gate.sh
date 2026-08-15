#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVMS_HOST_PORT="${OVMS_HOST_PORT:-8083}"
CHAINLIT_VERSION="${CHAINLIT_VERSION:-2.11.1}"
OVMS_EMBEDDING_MODEL="${OVMS_EMBEDDING_MODEL:-OpenVINO/Qwen3-Embedding-0.6B-int8-ov}"
OVMS_RERANKER_MODEL="${OVMS_RERANKER_MODEL:-OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov}"
OPENVINO_VECTOR_INDEX="${OPENVINO_VECTOR_INDEX:-ops-knowledge-openvino-v1}"

printf '\n[1/6] Static + architecture + deterministic grounding preflight\n'
uv run python -m py_compile \
  src/rag_ops_guard/agent/semantic_gate.py \
  src/rag_ops_guard/agent/grounding.py \
  src/rag_ops_guard/agent/react_agent.py \
  src/rag_ops_guard/retrieval/hybrid.py \
  src/rag_ops_guard/retrieval/resilient.py \
  scripts/semantic_gate_smoke.py \
  scripts/react_rag_smoke.py \
  scripts/chainlit_react_ui.py
uv run ruff check \
  src/rag_ops_guard/agent/semantic_gate.py \
  src/rag_ops_guard/agent/grounding.py \
  src/rag_ops_guard/agent/react_agent.py \
  src/rag_ops_guard/retrieval/hybrid.py \
  src/rag_ops_guard/retrieval/resilient.py \
  tests/architecture/test_grounding_policy_guard.py \
  tests/unit/test_semantic_gate.py \
  tests/unit/test_grounding_policy.py \
  tests/unit/test_react_grounding_tools.py \
  tests/unit/test_resilient_retrieval.py
uv run mypy \
  src/rag_ops_guard/agent/semantic_gate.py \
  src/rag_ops_guard/agent/grounding.py \
  src/rag_ops_guard/agent/react_agent.py \
  src/rag_ops_guard/retrieval/hybrid.py \
  src/rag_ops_guard/retrieval/resilient.py
uv run pytest \
  tests/architecture/test_grounding_policy_guard.py \
  tests/unit/test_react_streaming.py \
  tests/unit/test_semantic_gate.py \
  tests/unit/test_grounding_policy.py \
  tests/unit/test_react_grounding_tools.py \
  tests/unit/test_resilient_retrieval.py \
  -q

printf '\n[2/6] Local runtime + corpus integrity\n'
make generation-model local-core-up openvino-up

export EMBEDDING_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export EMBEDDING_MODEL="$OVMS_EMBEDDING_MODEL"
export RERANKER_BASE_URL="http://127.0.0.1:${OVMS_HOST_PORT}/v3"
export RERANKER_MODEL="$OVMS_RERANKER_MODEL"
export S3_VECTOR_INDEX="$OPENVINO_VECTOR_INDEX"
export RETRIEVAL_TOP_K="${BETA_RETRIEVAL_TOP_K:-8}"
export RETRIEVAL_CONTEXT_K="${BETA_RETRIEVAL_CONTEXT_K:-3}"
uv run python scripts/local/ensure_data.py

printf '\n[3/6] Learned semantic gate on OpenVINO cross-encoder\n'
uv run python scripts/semantic_gate_smoke.py

printf '\n[4/6] Direct chat/code streaming\n'
uv run python scripts/react_direct_ui_smoke.py

printf '\n[5/6] Conversational Grounding v5: learned gate + deterministic retrieval\n'
uv run python scripts/react_rag_smoke.py

printf '\n[6/6] Chainlit application import/config\n'
uv run --with "chainlit==${CHAINLIT_VERSION}" python -c \
  'import runpy; runpy.run_path("scripts/chainlit_react_ui.py", run_name="chainlit_migration_gate")'

cat <<'EOF'

CHAINLIT MIGRATION GATE: HEADLESS PASS

Conversational Grounding v5 invariants:
  - no generative model is used for conversational routing
  - routing uses the existing learned OpenVINO cross-encoder, not embedding cosine prototypes
  - dialogue/entity catalog budget = 0
  - regex dialogue routing budget = 0
  - one abstract policy hypothesis is allowed per control action
  - uncertain semantic decisions fail closed to retrieval
  - retrieval query construction is deterministic
  - the literal current user turn is always the reranker query
  - generation Qwen is used for answers, not routing policy
  - evidence/context/TTL/rollback remain deterministic

Manual UI confirmation still required before merge:
  1. make chainlit-beta
  2. open http://127.0.0.1:8001
  3. direct code streams with policy=direct and no RAG
  4. Calypso retries answers 3 and exposes source cards
  5. a contextual new-fact follow-up re-grounds and answers from evidence
  6. a pure transform uses visible conversation history without another RAG call
  7. ask about a previously unseen internal entity and verify unsupported evidence abstains
  8. change environment and verify stale evidence is not used as factual support
  9. Stop interrupts a generation without corrupting prior message/grounding state
  10. Retry works after a recoverable failure
  11. Local Status behaves correctly

If those pass, v5 is ready for the next promotion decision.
EOF
