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


def test_ingestion_is_idempotent() -> None:
    objects = FakeObjectStore({"raw/policy.md": DOCUMENT})
    vectors = FakeVectorStore()
    service = IngestionService(
        object_store=objects,
        vector_store=vectors,
        embeddings=FakeEmbeddingProvider(),
        chunker=MarkdownChunker(lambda text: len(text.split()), 400, 60),
    )
    first = service.ingest("raw/policy.md")
    second = service.ingest("raw/policy.md")
    assert first.status == "ingested"
    assert second.status == "no_op"
    assert first.chunks == second.chunks
    assert len(vectors.stored) == first.chunks
