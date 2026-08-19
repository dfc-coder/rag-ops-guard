"""SPEC-P4-MULTITENANCY 4.3 red contracts for authenticated runtime storage isolation."""

import pytest

from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.tenancy import KeyLayout, RequestContext
from rag_ops_guard.tenancy.runtime_auth import MissingApiKeyError, request_context_from_event
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeVectorStore


class FakeAuthenticator:
    def __init__(self) -> None:
        self.seen: str | None = None

    def authenticate(self, api_key: str) -> RequestContext:
        self.seen = api_key
        return RequestContext(principal="api-key:key-a", tenant_id="tenant-a")


def test_transport_auth_extracts_api_key_case_insensitively_and_body_is_not_authority() -> None:
    authenticator = FakeAuthenticator()
    event = {
        "headers": {"X-Api-Key": "key-a.secret"},
        "body": '{"question":"hello","tenant_id":"tenant-b"}',
    }

    context = request_context_from_event(event, authenticator=authenticator)

    assert authenticator.seen == "key-a.secret"
    assert context.tenant_id == "tenant-a"


def test_transport_auth_rejects_missing_api_key() -> None:
    with pytest.raises(MissingApiKeyError):
        request_context_from_event({"headers": {}}, authenticator=FakeAuthenticator())


def test_ingestion_rejects_legacy_and_cross_tenant_source_keys_before_read() -> None:
    layout = KeyLayout("tenant-a")
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
        service.ingest("t/tenant-b/raw/api/payments.md")


def test_ingestion_accepts_only_authenticated_tenant_raw_prefix() -> None:
    layout = KeyLayout("tenant-a")
    key = layout.raw_key("api/payments.md")
    objects = FakeObjectStore(
        {key: "# Payments API\n\nTenant A documentation."}
    )
    service = IngestionService(
        object_store=objects,
        vector_store=FakeVectorStore(),
        embeddings=FakeEmbeddingProvider(),
        chunker=MarkdownChunker(lambda text: len(text.split()), 400, 60),
        key_layout=layout,
    )

    response = service.ingest(key)

    assert response.status == "ingested"
    assert objects.list_keys(layout.chunks_prefix)
