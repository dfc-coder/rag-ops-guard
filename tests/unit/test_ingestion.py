from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.tenancy import KeyLayout
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeVectorStore

DOCUMENT = """---
id: payment-retry-policy-v2
logical_id: payment-retry-policy
title: Payment Retry Policy
version: "2.0"
status: active
effective_date: 2026-06-01
system: payments
environment: production
document_type: runbook
authority: 100
supersedes:
  - payment-retry-policy-v1
---
# Timeout Handling
Retry a Calypso timeout at most three times. Verify transaction status before manual replay.
"""

DOCUMENT_TWO_SECTIONS = """---
id: payment-retry-policy-v2
logical_id: payment-retry-policy
title: Payment Retry Policy
version: "2.0"
status: active
effective_date: 2026-06-01
system: payments
environment: production
document_type: runbook
authority: 100
supersedes:
  - payment-retry-policy-v1
---
# Timeout Handling
Retry a Calypso timeout at most three times.

# Escalation
Escalate to Treasury Integrations after the third failure.
"""

_LAYOUT = KeyLayout("default")


def _service(objects: FakeObjectStore, vectors: FakeVectorStore) -> IngestionService:
    return IngestionService(
        object_store=objects,
        vector_store=vectors,
        embeddings=FakeEmbeddingProvider(),
        chunker=MarkdownChunker(lambda text: len(text.split()), 400, 60),
        key_layout=_LAYOUT,
    )


def test_ingestion_is_idempotent() -> None:
    objects = FakeObjectStore({"raw/policy.md": DOCUMENT})
    vectors = FakeVectorStore()
    service = _service(objects, vectors)
    first = service.ingest("raw/policy.md")
    second = service.ingest("raw/policy.md")
    assert first.status == "ingested"
    assert second.status == "no_op"
    assert first.chunks == second.chunks
    assert len(vectors.stored) == first.chunks


def test_plain_markdown_without_frontmatter_flows_through_service() -> None:
    """SPEC-2.1"""
    objects = FakeObjectStore(
        {"raw/architecture-notes.md": "# Architecture Notes\n\nGeneric content."}
    )
    vectors = FakeVectorStore()

    result = _service(objects, vectors).ingest("raw/architecture-notes.md")

    assert result.status == "ingested"
    assert result.logical_id.startswith("doc-")
    assert result.version == "1"
    assert result.chunks == 1
    assert len(vectors.stored) == 1


def test_plain_text_without_frontmatter_flows_through_service() -> None:
    """SPEC-2.2"""
    objects = FakeObjectStore({"raw/manual-operativo.txt": "Texto operativo general."})
    vectors = FakeVectorStore()

    result = _service(objects, vectors).ingest("raw/manual-operativo.txt")

    assert result.status == "ingested"
    assert result.chunks == 1


def test_changed_document_deletes_stale_chunk_objects() -> None:
    objects = FakeObjectStore({"raw/policy.md": DOCUMENT_TWO_SECTIONS})
    vectors = FakeVectorStore()
    service = _service(objects, vectors)

    first = service.ingest("raw/policy.md")
    assert first.chunks == 2
    chunk_prefix = _LAYOUT.chunk_prefix("payment-retry-policy", "2.0")
    old_keys = objects.list_keys(chunk_prefix)
    assert len(old_keys) == 2

    objects.put_text("raw/policy.md", DOCUMENT)
    second = service.ingest("raw/policy.md")

    assert second.status == "ingested"
    assert second.chunks == 1
    current_keys = objects.list_keys(chunk_prefix)
    assert len(current_keys) == 1
    assert old_keys[1] in objects.deleted
