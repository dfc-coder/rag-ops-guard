#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHAINLIT_HOST_PORT="${CHAINLIT_HOST_PORT:-8001}"
CHAINLIT_PACKAGE_VERSION="${CHAINLIT_VERSION:-2.11.1}"
unset CHAINLIT_VERSION

if [[ ! -s .local/api-url ]]; then
  echo 'Canonical API is not provisioned. Run: make up' >&2
  exit 2
fi

RAG_API_URL="$(cat .local/api-url)"
export RAG_API_URL

echo
printf 'RAG Ops Guard · Chainlit: http://127.0.0.1:%s\n' "$CHAINLIT_HOST_PORT"
printf 'Canonical API: %s/v1/query\n' "$RAG_API_URL"
echo 'Path: Chainlit -> Floci API Gateway -> Lambda -> Agent -> Qwen/Retrieval'
echo 'Use Ctrl+C to stop only the UI. Stop the runtime with: make down'
echo

exec uv run --with "chainlit==${CHAINLIT_PACKAGE_VERSION}" \
  chainlit run scripts/chainlit_api_ui.py \
  --host 127.0.0.1 \
  --port "$CHAINLIT_HOST_PORT" \
  --headless
