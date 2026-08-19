from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ops_guard.control_plane import (
    APPCONFIG_APPLICATION,
    APPCONFIG_ENVIRONMENT,
    APPCONFIG_PROFILE,
    ControlPlaneConfig,
    ControlPlaneResolutionError,
    fetch_control_plane,
)

ROOT = Path(__file__).resolve().parents[2]


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeAppConfigData:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.started_with: dict[str, object] | None = None
        self.latest_with: dict[str, object] | None = None

    def start_configuration_session(self, **kwargs: object) -> dict[str, object]:
        self.started_with = dict(kwargs)
        return {"InitialConfigurationToken": "token-1"}

    def get_latest_configuration(self, **kwargs: object) -> dict[str, object]:
        self.latest_with = dict(kwargs)
        return {
            "Configuration": _Body(json.dumps(self.payload).encode("utf-8")),
            "NextPollConfigurationToken": "token-2",
            "NextPollIntervalInSeconds": 45,
        }


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "resources": {
            "config_table": "rag-ops-config",
            "tenant_credential_table": "rag-ops-tenants",
            "document_bucket": "docs",
            "vector_bucket": "vectors",
            "vector_index_base": "ops-knowledge-openvino-v1",
        },
        "services": {
            "llm": {"base_url": "http://llama-gen:8080/v1", "model": "qwen"},
            "embedding": {
                "base_url": "http://rag-ops-ovms-rag:8000/v3",
                "model": "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
                "dimension": 1024,
            },
            "reranker": {
                "base_url": "http://rag-ops-ovms-rag:8000/v3",
                "model": "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov",
            },
        },
        "secret_refs": {},
        "bootstrap": {
            "config_head_ttl_seconds": 45,
            "config_max_stale_seconds": 300,
            "fail_closed": True,
        },
    }


def test_fetch_control_plane_uses_canonical_appconfigdata_session() -> None:
    client = _FakeAppConfigData(_payload())

    resolved = fetch_control_plane(client=client)

    assert resolved == ControlPlaneConfig.model_validate(_payload())
    assert client.started_with == {
        "ApplicationIdentifier": APPCONFIG_APPLICATION,
        "EnvironmentIdentifier": APPCONFIG_ENVIRONMENT,
        "ConfigurationProfileIdentifier": APPCONFIG_PROFILE,
    }
    assert client.latest_with == {"ConfigurationToken": "token-1"}


def test_application_resolver_never_overrides_aws_endpoint() -> None:
    source = (ROOT / "src/rag_ops_guard/control_plane.py").read_text(encoding="utf-8")

    assert 'boto3.client("appconfigdata")' in source
    assert "endpoint_url=" not in source
    assert "AWS_ENDPOINT_URL" not in source
    assert "os.getenv" not in source
    assert "os.environ" not in source


def test_empty_initial_configuration_fails_closed() -> None:
    client = _FakeAppConfigData(_payload())

    def empty_latest(**kwargs: object) -> dict[str, object]:
        client.latest_with = dict(kwargs)
        return {"Configuration": _Body(b""), "NextPollConfigurationToken": "token-2"}

    client.get_latest_configuration = empty_latest  # type: ignore[method-assign]

    with pytest.raises(ControlPlaneResolutionError, match="empty"):
        fetch_control_plane(client=client)
