from __future__ import annotations

import pytest

from rag_ops_guard.control_plane import fetch_control_plane

pytestmark = pytest.mark.integration


def test_floci_serves_canonical_appconfig_control_plane_through_application_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Platform/test-runner concern only: application code still relies exclusively on the SDK provider chain.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

    resolved = fetch_control_plane()

    assert resolved.schema_version == 1
    assert resolved.resources.config_table == "rag-ops-config"
    assert resolved.resources.tenant_credential_table == "rag-ops-tenants"
    assert resolved.resources.document_bucket == "rag-ops-guard-docs-local"
    assert resolved.resources.vector_bucket == "rag-ops-guard-vectors-local"
    assert resolved.resources.vector_index_base == "ops-knowledge-openvino-v1"
    assert resolved.services.embedding.dimension == 1024
    assert resolved.bootstrap.fail_closed is True
