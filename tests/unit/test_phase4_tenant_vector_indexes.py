"""SPEC-P4-MULTITENANCY 4.4 red contracts for structural vector isolation."""

from rag_ops_guard.tenancy import KeyLayout


def test_tenants_never_share_vector_index_names() -> None:
    base = "ops-knowledge-openvino-v1"
    tenant_a = KeyLayout("tenant-a").vector_index(base)
    tenant_b = KeyLayout("tenant-b").vector_index(base)

    assert tenant_a == "ops-knowledge-openvino-v1--tenant-a"
    assert tenant_b == "ops-knowledge-openvino-v1--tenant-b"
    assert tenant_a != tenant_b


def test_vector_index_identity_does_not_depend_on_metadata_filters() -> None:
    base = "ops-knowledge-openvino-v1"
    assert KeyLayout("tenant-a").vector_index(base).endswith("--tenant-a")
    assert KeyLayout("tenant-b").vector_index(base).endswith("--tenant-b")
