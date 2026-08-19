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


class _FakeAppConfig:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_applications(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("list_applications", dict(kwargs)))
        return {"Items": [{"Id": "app-123", "Name": APPCONFIG_APPLICATION}]}

    def list_environments(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("list_environments", dict(kwargs)))
        return {"Items": [{"Id": "env-456", "Name": APPCONFIG_ENVIRONMENT}]}

    def list_configuration_profiles(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("list_configuration_profiles", dict(kwargs)))
        return {"Items": [{"Id": "prof-789", "Name": APPCONFIG_PROFILE}]}


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


def test_fetch_control_plane_resolves_names_then_uses_physical_ids() -> None:
    management = _FakeAppConfig()
    data = _FakeAppConfigData(_payload())

    resolved = fetch_control_plane(management_client=management, data_client=data)

    assert resolved == ControlPlaneConfig.model_validate(_payload())
    assert management.calls == [
        ("list_applications", {}),
        ("list_environments", {"ApplicationId": "app-123"}),
        ("list_configuration_profiles", {"ApplicationId": "app-123"}),
    ]
    assert data.started_with == {
        "ApplicationIdentifier": "app-123",
        "EnvironmentIdentifier": "env-456",
        "ConfigurationProfileIdentifier": "prof-789",
    }
    assert data.latest_with == {"ConfigurationToken": "token-1"}


def test_application_resolver_uses_standard_sdk_provider_chain_only() -> None:
    source = (ROOT / "src/rag_ops_guard/control_plane.py").read_text(encoding="utf-8")

    assert 'boto3.client("appconfig")' in source
    assert 'boto3.client("appconfigdata")' in source
    assert "endpoint_url=" not in source
    assert "AWS_ENDPOINT_URL" not in source
    assert "os.getenv" not in source
    assert "os.environ" not in source
    assert APPCONFIG_APPLICATION in source
    assert APPCONFIG_ENVIRONMENT in source
    assert APPCONFIG_PROFILE in source


def test_missing_named_control_plane_resource_fails_closed() -> None:
    management = _FakeAppConfig()

    def no_environments(**kwargs: object) -> dict[str, object]:
        management.calls.append(("list_environments", dict(kwargs)))
        return {"Items": []}

    management.list_environments = no_environments  # type: ignore[method-assign]

    with pytest.raises(ControlPlaneResolutionError, match="runtime"):
        fetch_control_plane(
            management_client=management,
            data_client=_FakeAppConfigData(_payload()),
        )


def test_empty_initial_configuration_fails_closed() -> None:
    management = _FakeAppConfig()
    data = _FakeAppConfigData(_payload())

    def empty_latest(**kwargs: object) -> dict[str, object]:
        data.latest_with = dict(kwargs)
        return {"Configuration": _Body(b""), "NextPollConfigurationToken": "token-2"}

    data.get_latest_configuration = empty_latest  # type: ignore[method-assign]

    with pytest.raises(ControlPlaneResolutionError, match="empty"):
        fetch_control_plane(management_client=management, data_client=data)
