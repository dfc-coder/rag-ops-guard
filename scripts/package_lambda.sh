#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/.local/lambda-package"
LAMBDA_PYTHON_VERSION="${LAMBDA_PYTHON_VERSION:-3.12}"
rm -rf "$TARGET"
mkdir -p "$TARGET"
cd "$ROOT"
uv pip install --python "$LAMBDA_PYTHON_VERSION" --target "$TARGET" .
echo "$TARGET"
