from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
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


def _service(objects: FakeObjectStore, vectors: FakeVectorStore) -> IngestionService:
    return IngestionService(
        object_store=objects,
        vector_store=vectors,
        embeddings=FakeEmbeddingProvider(),
        chunker=MarkdownChunker(lambda text: len(text.split()), 400, 60),
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


def test_changed_document_deletes_stale_chunk_objects() -> None:
    objects = FakeObjectStore({"raw/policy.md": DOCUMENT_TWO_SECTIONS})
    vectors = FakeVectorStore()
    service = _service(objects, vectors)

    first = service.ingest("raw/policy.md")
    assert first.chunks == 2
    old_keys = objects.list_keys("chunks/payment-retry-policy/2.0/")
    assert len(old_keys) == 2

    objects.put_text("raw/policy.md", DOCUMENT)
    second = service.ingest("raw/policy.md")

    assert second.status == "ingested"
    assert second.chunks == 1
    current_keys = objects.list_keys("chunks/payment-retry-policy/2.0/")
    assert len(current_keys) == 1
    assert old_keys[1] in objects.deleted
