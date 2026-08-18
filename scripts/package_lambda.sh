#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.local/lambda-build"
ZIP_PATH="$ROOT/.local/lambda-package.zip"
LAMBDA_PYTHON_VERSION="${LAMBDA_PYTHON_VERSION:-3.12}"

rm -rf "$BUILD_DIR" "$ZIP_PATH"
mkdir -p "$BUILD_DIR"
cd "$ROOT"

uv pip install --python "$LAMBDA_PYTHON_VERSION" --target "$BUILD_DIR" .

# Phase 2 fallback: embed the last trusted, non-secret effective snapshot into the immutable
# Lambda artifact. DynamoDB and S3 remain preferred; this copy is used only after both are absent.
if [[ -f "$ROOT/.local/config-snapshot.json" ]]; then
  mkdir -p "$BUILD_DIR/rag_ops_guard/configstore"
  cp "$ROOT/.local/config-snapshot.json" \
    "$BUILD_DIR/rag_ops_guard/configstore/baked_snapshot.json"
fi

LAMBDA_BUILD_DIR="$BUILD_DIR" LAMBDA_ZIP_PATH="$ZIP_PATH" \
uv run --no-project --python "$LAMBDA_PYTHON_VERSION" python - <<'PY'
from __future__ import annotations

import os
import zipfile
from pathlib import Path

build_dir = Path(os.environ["LAMBDA_BUILD_DIR"])
zip_path = Path(os.environ["LAMBDA_ZIP_PATH"])

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(build_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(build_dir).as_posix()
        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

print(zip_path)
PY
