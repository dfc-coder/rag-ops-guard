"""SPEC-P4-MULTITENANCY 4.0 red contracts for tenant key layout."""

import pytest

from rag_ops_guard.tenancy.key_layout import KeyLayout


def test_key_layout_owns_all_tenant_storage_paths() -> None:
    layout = KeyLayout("tenant-a")

    assert layout.tenant_prefix == "t/tenant-a/"
    assert layout.raw_key("api/payments.md") == "t/tenant-a/raw/api/payments.md"
    assert layout.chunks_prefix == "t/tenant-a/chunks/"
    assert layout.chunk_prefix("payments", "2.0") == "t/tenant-a/chunks/payments/2.0/"
    assert layout.chunk_key("payments", "2.0", 7) == (
        "t/tenant-a/chunks/payments/2.0/chunk-007.json"
    )
    assert layout.manifest_key("payments", "2.0") == "t/tenant-a/manifests/payments/2.0.json"


def test_key_layout_rejects_cross_tenant_and_legacy_ingest_keys() -> None:
    layout = KeyLayout("tenant-a")

    assert layout.require_ingest_key("t/tenant-a/raw/api/payments.md") == (
        "t/tenant-a/raw/api/payments.md"
    )
    with pytest.raises(ValueError, match="tenant-scoped raw key"):
        layout.require_ingest_key("raw/api/payments.md")
    with pytest.raises(ValueError, match="tenant-scoped raw key"):
        layout.require_ingest_key("t/tenant-b/raw/api/payments.md")


def test_key_layout_rejects_unsafe_tenant_and_relative_paths() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        KeyLayout("../tenant-a")

    layout = KeyLayout("tenant-a")
    for unsafe in ("/absolute.md", "../escape.md", "api/../../escape.md", ""):
        with pytest.raises(ValueError):
            layout.raw_key(unsafe)


def test_vector_index_name_is_deterministic_and_tenant_specific() -> None:
    tenant_a = KeyLayout("tenant-a")
    tenant_b = KeyLayout("tenant-b")

    assert tenant_a.vector_index("ops-knowledge-openvino-v1") == (
        "ops-knowledge-openvino-v1--tenant-a"
    )
    assert tenant_a.vector_index("ops-knowledge-openvino-v1") != tenant_b.vector_index(
        "ops-knowledge-openvino-v1"
    )
