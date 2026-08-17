from __future__ import annotations

import importlib.util
import io
import json
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


def test_floci_execution_endpoint_uses_real_api_id_and_path_style_data_plane() -> None:
    provision = _load_provision()
    assert provision.floci_execution_endpoint("http://localhost:4566", "api-123") == (
        "http://localhost:4566/execute-api/api-123/$default"
    )


def test_lambda_packaging_python_is_pinned_to_runtime() -> None:
    provision = _load_provision()
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    package_script = (ROOT / "scripts/package_lambda.sh").read_text(encoding="utf-8")

    assert provision.LAMBDA_PYTHON_VERSION == "3.12"
    assert "LAMBDA_PYTHON_VERSION ?= 3.12" in makefile
    assert 'LAMBDA_PYTHON_VERSION="${LAMBDA_PYTHON_VERSION:-3.12}"' in package_script
    assert 'uv pip install --python "$LAMBDA_PYTHON_VERSION" --target "$BUILD_DIR" .' in package_script
    assert ".local/lambda-package.zip" in package_script


def test_api_probe_accepts_lambda_invalid_request_response() -> None:
    provision = _load_provision()
    response = httpx.Response(
        400,
        headers={"content-type": "application/json"},
        json={"error": "invalid_request", "detail": "missing field"},
    )
    provision.validate_api_probe(response, route="/v1/ingest")


def test_api_probe_rejects_s3_xml_misrouting() -> None:
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
        provision.validate_api_probe(response, route="/v1/ingest")


def test_direct_lambda_probe_surfaces_function_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provision = _load_provision()

    class FakeLambda:
        def invoke(self, **_: object) -> dict[str, object]:
            return {
                "FunctionError": "Unhandled",
                "Payload": io.BytesIO(b'{"errorMessage":"ImportModuleError"}'),
            }

    monkeypatch.setattr(provision, "client", lambda service, **kwargs: FakeLambda())
    with pytest.raises(RuntimeError, match="ImportModuleError"):
        provision.probe_lambda("rag-ops-guard-ingest")


def test_direct_lambda_probe_accepts_invalid_request_proxy_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision = _load_provision()
    proxy_payload = {
        "statusCode": 400,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"error": "invalid_request", "detail": "missing field"}),
    }

    class FakeLambda:
        def invoke(self, **_: object) -> dict[str, object]:
            return {"Payload": io.BytesIO(json.dumps(proxy_payload).encode())}

    monkeypatch.setattr(provision, "client", lambda service, **kwargs: FakeLambda())
    provision.probe_lambda("rag-ops-guard-ingest")


def test_physical_provisioning_uses_no_magic_hot_reload_or_forced_api_id() -> None:
    provision = (ROOT / "scripts/local/provision.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker/docker-compose.yml").read_text(encoding="utf-8")

    assert '"hot-reload"' not in provision
    assert "floci:override-id" not in provision
    assert "FLOCI_SERVICES_LAMBDA_HOT_RELOAD_ENABLED" not in compose
    assert 'Code={"S3Bucket": LAMBDA_CODE_BUCKET, "S3Key": code_key}' in provision


def test_lambda_environment_propagates_langsmith_only_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision = _load_provision()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "golden-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-test-key")
    monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "workspace-test")

    env = provision.lambda_environment()

    assert env["LANGSMITH_TRACING"] == "true"
    assert env["LANGSMITH_PROJECT"] == "golden-test"
    assert env["LANGSMITH_API_KEY"] == "secret-test-key"
    assert env["LANGSMITH_WORKSPACE_ID"] == "workspace-test"

    monkeypatch.delenv("LANGSMITH_API_KEY")
    assert provision.lambda_environment()["LANGSMITH_TRACING"] == "false"
