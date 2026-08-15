#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVMS_HOST_PORT="${OVMS_HOST_PORT:-8083}"
CHAINLIT_HOST_PORT="${CHAINLIT_HOST_PORT:-8001}"
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

uv run python scripts/local/ensure_data.py

echo
printf 'Chainlit RAG Ops Guard: http://127.0.0.1:%s\n' "$CHAINLIT_HOST_PORT"
printf 'Chainlit version: %s\n' "$CHAINLIT_VERSION"
echo 'Use Ctrl+C to stop the Chainlit UI. Shared local services can be stopped with: make local-down'
echo

exec uv run --with "chainlit==${CHAINLIT_VERSION}" \
  chainlit run scripts/chainlit_react_ui.py \
  --host 127.0.0.1 \
  --port "$CHAINLIT_HOST_PORT" \
  --headless
