from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_provision() -> ModuleType:
    path = ROOT / "scripts/local/provision.py"
    spec = importlib.util.spec_from_file_location("rag_ops_local_provision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_floci_execution_endpoint_uses_path_style_data_plane() -> None:
    provision = _load_provision()

    assert provision.floci_execution_endpoint("http://localhost:4566", "api-123") == (
        "http://localhost:4566/execute-api/api-123/$default"
    )


def test_lambda_packaging_python_is_pinned_to_runtime() -> None:
    provision = _load_provision()
    package_script = (ROOT / "scripts/package_lambda.sh").read_text(encoding="utf-8")

    assert provision.LAMBDA_PYTHON_VERSION == "3.12"
    assert 'LAMBDA_PYTHON_VERSION="${LAMBDA_PYTHON_VERSION:-3.12}"' in package_script
    assert 'uv pip install --python "$LAMBDA_PYTHON_VERSION" --target "$TARGET" .' in package_script


def test_ingest_probe_accepts_lambda_invalid_request_response() -> None:
    provision = _load_provision()
    response = httpx.Response(
        400,
        headers={"content-type": "application/json"},
        json={"error": "invalid_request", "detail": "missing s3_key"},
    )

    provision.validate_ingest_probe(response)


def test_ingest_probe_rejects_s3_xml_misrouting() -> None:
    provision = _load_provision()
    response = httpx.Response(
        400,
        headers={"content-type": "application/xml"},
        text=(
            '<?xml version="1.0"?><Error><Code>InvalidArgument</Code>'
            "<Message>POST requires either ?uploads or ?uploadId.</Message></Error>"
        ),
    )

    with pytest.raises(RuntimeError, match="API Gateway data-plane probe failed"):
        provision.validate_ingest_probe(response)
