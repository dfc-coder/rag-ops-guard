#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/.local/lambda-package"
rm -rf "$TARGET"
mkdir -p "$TARGET"
cd "$ROOT"
uv pip install --target "$TARGET" .
echo "$TARGET"
