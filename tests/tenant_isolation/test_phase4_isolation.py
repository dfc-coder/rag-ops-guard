"""Blocking SPEC-P4-MULTITENANCY 4.6 tenant-isolation acceptance contracts."""

from __future__ import annotations

from typing import Any

import pytest

from rag_ops_guard.adapters.aws import s3_vectors
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from rag_ops_guard.configstore import tenant_store
from rag_ops_guard.configstore.envelope import (
    EnvelopeAuthenticationError,
    decrypt_secret,
    encrypt_secret,
)
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.container import Container
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.tenancy import KeyLayout, RequestContext
from rag_ops_guard.tenancy.runtime_auth import request_context_from_event
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeVectorStore

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
SAME_CONFIG_HASH = "a" * 64


def _context(tenant_id: str) -> RequestContext:
    return RequestContext(principal=f"api-key:key-{tenant_id}", tenant_id=tenant_id)


def test_chunks_and_object_keys_have_zero_cross_tenant_overlap() -> None:
    a = KeyLayout(TENANT_A)
    b = KeyLayout(TENANT_B)

    a_keys = {
        a.raw_key("api/payments.md"),
        a.chunk_key("payments", "1", 0),
        a.manifest_key("payments", "1"),
    }
    b_keys = {
        b.raw_key("api/payments.md"),
        b.chunk_key("payments", "1", 0),
        b.manifest_key("payments", "1"),
    }

    assert a_keys.isdisjoint(b_keys)
    assert all(a.owns(key) and not b.owns(key) for key in a_keys)
    assert all(b.owns(key) and not a.owns(key) for key in b_keys)


def test_ingestion_rejects_unprefixed_and_wrong_tenant_keys_before_object_read() -> None:
    layout = KeyLayout(TENANT_A)
    objects = FakeObjectStore()
    service = IngestionService(
        object_store=objects,
        vector_store=FakeVectorStore(),
        embeddings=FakeEmbeddingProvider(),
        chunker=MarkdownChunker(lambda text: len(text.split()), 400, 60),
        key_layout=layout,
    )

    with pytest.raises(ValueError, match="tenant-scoped raw key"):
        service.ingest("raw/api/payments.md")
    with pytest.raises(ValueError, match="tenant-scoped raw key"):
        service.ingest(KeyLayout(TENANT_B).raw_key("api/payments.md"))

    assert objects.values == {}


class _FakeDynamoClient:
    def __init__(self) -> None:
        self.keys: list[dict[str, Any]] = []

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.keys.append(kwargs["Key"])
        return {}


def test_effective_config_records_use_disjoint_tenant_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDynamoClient()
    monkeypatch.setattr(tenant_store.boto3, "client", lambda *args, **kwargs: fake)

    stores = [
        TenantDynamoDbConfigStore(
            endpoint_url="http://floci",
            region="us-east-1",
            access_key="test",
            secret_key="test",
            table="rag-ops-config",
            tenant_id=tenant_id,
        )
        for tenant_id in (TENANT_A, TENANT_B)
    ]
    for store in stores:
        assert store.get_head() is None

    assert fake.keys == [
        {"PK": {"S": "TENANT#tenant-a"}, "SK": {"S": "HEAD"}},
        {"PK": {"S": "TENANT#tenant-b"}, "SK": {"S": "HEAD"}},
    ]


def test_warm_dependency_container_never_crosses_tenants_with_same_config_hash() -> None:
    container: Container[dict[str, str]] = Container(max_generations=32)

    def factory(context: RequestContext, config_hash: str) -> dict[str, str]:
        return {"tenant_id": context.tenant_id, "config_hash": config_hash}

    tenant_a = container.get(_context(TENANT_A), SAME_CONFIG_HASH, factory)
    tenant_b = container.get(_context(TENANT_B), SAME_CONFIG_HASH, factory)

    assert tenant_a is not tenant_b
    assert tenant_a["tenant_id"] == TENANT_A
    assert tenant_b["tenant_id"] == TENANT_B
    assert set(container.keys()) == {
        (TENANT_A, SAME_CONFIG_HASH),
        (TENANT_B, SAME_CONFIG_HASH),
    }


def test_phase3_secret_aad_rejects_cross_tenant_decryption() -> None:
    dek = bytes(range(32))
    envelope = encrypt_secret(
        b"tenant-a-secret",
        dek=dek,
        scope="TENANT#tenant-a",
        key_name="provider_api_key",
        dek_version=7,
    )

    assert decrypt_secret(
        envelope,
        dek=dek,
        scope="TENANT#tenant-a",
        key_name="provider_api_key",
    ) == b"tenant-a-secret"
    with pytest.raises(EnvelopeAuthenticationError):
        decrypt_secret(
            envelope,
            dek=dek,
            scope="TENANT#tenant-b",
            key_name="provider_api_key",
        )


class _FakeVectorClient:
    pass


def test_vector_index_selection_is_structurally_tenant_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s3_vectors.boto3, "client", lambda *args, **kwargs: _FakeVectorClient())
    base = "ops-knowledge-openvino-v1"
    objects = FakeObjectStore()
    layout_a = KeyLayout(TENANT_A)
    layout_b = KeyLayout(TENANT_B)

    store_a = S3VectorsStore(
        "vectors",
        layout_a.vector_index(base),
        objects,
        "http://floci",
        "us-east-1",
        "test",
        "test",
        layout_a,
    )
    store_b = S3VectorsStore(
        "vectors",
        layout_b.vector_index(base),
        objects,
        "http://floci",
        "us-east-1",
        "test",
        "test",
        layout_b,
    )

    assert store_a._index == "ops-knowledge-openvino-v1--tenant-a"
    assert store_b._index == "ops-knowledge-openvino-v1--tenant-b"
    assert store_a._index != store_b._index


class _AuthenticatorA:
    def authenticate(self, api_key: str) -> RequestContext:
        assert api_key == "key-a.secret-a"
        return _context(TENANT_A)


def test_request_body_cannot_override_authenticated_tenant() -> None:
    event = {
        "headers": {"x-api-key": "key-a.secret-a"},
        "body": '{"question":"hello","tenant_id":"tenant-b"}',
    }

    context = request_context_from_event(event, authenticator=_AuthenticatorA())

    assert context.tenant_id == TENANT_A
    assert context.tenant_id != TENANT_B
