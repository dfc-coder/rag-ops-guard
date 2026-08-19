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


def test_lambda_packaging_python_and_dependencies_are_canonical() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    package_script = (ROOT / "scripts/package_lambda.sh").read_text(encoding="utf-8")

    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.13"
    assert "PYTHON_VERSION := $(strip $(shell cat .python-version))" in makefile
    assert "LAMBDA_PYTHON_VERSION ?= $(PYTHON_VERSION)" in makefile
    assert 'PYTHON_VERSION_FILE="$ROOT/.python-version"' in package_script
    assert "uv export" in package_script
    assert "--frozen" in package_script
    assert "--no-emit-project" in package_script
    assert '--requirement "$REQUIREMENTS_PATH"' in package_script
    assert "uv build --wheel" in package_script
    assert "--no-deps" in package_script
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


def test_local_provisioner_does_not_duplicate_cdk_resources() -> None:
    provision = (ROOT / "scripts/local/provision.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker/docker-compose.yml").read_text(encoding="utf-8")

    assert 'client("cloudformation").describe_stacks' in provision
    assert "materialize_floci_s3_vectors" in provision
    assert "create_function" not in provision
    assert "create_api" not in provision
    assert "create_table" not in provision
    assert "create_bucket" not in provision
    assert '"hot-reload"' not in provision
    assert "floci:override-id" not in provision
    assert "FLOCI_SERVICES_LAMBDA_HOT_RELOAD_ENABLED" not in compose


def test_stack_outputs_require_declared_values(monkeypatch: pytest.MonkeyPatch) -> None:
    provision = _load_provision()

    class FakeCloudFormation:
        def describe_stacks(self, **_: object) -> dict[str, object]:
            return {
                "Stacks": [
                    {
                        "Outputs": [
                            {"OutputKey": "ApiId", "OutputValue": "api-123"},
                            {"OutputKey": "QueryFunctionName", "OutputValue": "query"},
                        ]
                    }
                ]
            }

    monkeypatch.setattr(provision, "client", lambda service, **kwargs: FakeCloudFormation())
    outputs = provision.stack_outputs()
    assert outputs["ApiId"] == "api-123"
    with pytest.raises(RuntimeError, match="IngestFunctionName"):
        provision.required_output(outputs, "IngestFunctionName")
