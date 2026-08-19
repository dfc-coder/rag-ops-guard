#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.local/lambda-build"
DIST_DIR="$ROOT/.local/lambda-dist"
REQUIREMENTS_PATH="$ROOT/.local/lambda-requirements.txt"
ZIP_PATH="$ROOT/.local/lambda-package.zip"
PYTHON_VERSION_FILE="$ROOT/.python-version"

if [[ ! -f "$PYTHON_VERSION_FILE" ]]; then
  echo "Missing $PYTHON_VERSION_FILE" >&2
  exit 2
fi

PROJECT_PYTHON_VERSION="$(tr -d '[:space:]' < "$PYTHON_VERSION_FILE")"
LAMBDA_PYTHON_VERSION="${LAMBDA_PYTHON_VERSION:-$PROJECT_PYTHON_VERSION}"

if [[ "$LAMBDA_PYTHON_VERSION" != "$PROJECT_PYTHON_VERSION" ]]; then
  echo "Lambda Python $LAMBDA_PYTHON_VERSION diverges from .python-version $PROJECT_PYTHON_VERSION" >&2
  exit 2
fi

rm -rf "$BUILD_DIR" "$DIST_DIR" "$REQUIREMENTS_PATH" "$ZIP_PATH"
mkdir -p "$BUILD_DIR" "$DIST_DIR" "$ROOT/.local"
cd "$ROOT"

# The deployed Lambda must use the same dependency graph validated by CI.
uv export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --format requirements.txt \
  --output-file "$REQUIREMENTS_PATH"
uv pip install \
  --python "$PROJECT_PYTHON_VERSION" \
  --target "$BUILD_DIR" \
  --requirement "$REQUIREMENTS_PATH"

# Install the first-party package separately so the lock controls third-party dependencies.
uv build --wheel --out-dir "$DIST_DIR"
WHEEL_PATH="$(find "$DIST_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)"
if [[ -z "$WHEEL_PATH" ]]; then
  echo "Lambda application wheel was not produced" >&2
  exit 2
fi
uv pip install \
  --python "$PROJECT_PYTHON_VERSION" \
  --target "$BUILD_DIR" \
  --no-deps \
  "$WHEEL_PATH"

# Phase 2 fallback: embed the last trusted, non-secret effective snapshot into the immutable
# Lambda artifact. DynamoDB and S3 remain preferred; this copy is used only after both are absent.
if [[ -f "$ROOT/.local/config-snapshot.json" ]]; then
  mkdir -p "$BUILD_DIR/rag_ops_guard/configstore"
  cp "$ROOT/.local/config-snapshot.json" \
    "$BUILD_DIR/rag_ops_guard/configstore/baked_snapshot.json"
fi

LAMBDA_BUILD_DIR="$BUILD_DIR" LAMBDA_ZIP_PATH="$ZIP_PATH" \
uv run --no-project --python "$PROJECT_PYTHON_VERSION" python - <<'PY'
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
