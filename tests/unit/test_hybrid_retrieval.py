from __future__ import annotations

from rag_ops_guard.domain.models import DocumentType, Evidence, QueryContext
from rag_ops_guard.retrieval.bm25 import BM25Index
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, reciprocal_rank_fusion
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeVectorStore


def _named_evidence(title: str, text: str, *, logical_id: str, distance: float = 0.4) -> Evidence:
    meta = metadata(doc_id=f"{logical_id}-v1", logical_id=logical_id)
    meta.title = title
    meta.document_type = DocumentType.RUNBOOK
    item = evidence(meta=meta, text=text, distance=distance)
    item.chunk.title = title
    return item


def test_bm25_recovers_exact_entity_document() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failures must be retried independently.",
        logical_id="sendgrid-failure",
    )
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
    )

    results = BM25Index([sendgrid, payments]).search("What does SendGrid do?", limit=2)

    assert results[0].chunk.logical_id == "sendgrid-failure"


def test_rrf_can_promote_lexical_result_missed_by_dense_top_rank() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failures must be retried independently.",
        logical_id="sendgrid-failure",
        distance=0.8,
    )
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.1,
    )

    fused = reciprocal_rank_fusion(
        dense=[payments, sendgrid],
        lexical=[sendgrid, payments],
    )

    assert {item.chunk.logical_id for item in fused[:2]} == {
        "sendgrid-failure",
        "payment-retry-policy",
    }


def test_hybrid_search_includes_lexical_document_missing_from_dense_results() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failures must be retried independently.",
        logical_id="sendgrid-failure",
    )
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.1,
    )
    objects = FakeObjectStore()
    for item in (sendgrid, payments):
        key = (
            f"chunks/{item.chunk.logical_id}/{item.chunk.version}/"
            f"chunk-{item.chunk.chunk_index:03d}.json"
        )
        objects.put_text(key, item.chunk.model_dump_json())

    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[payments]),
        objects=objects,
        resolver=EvidenceResolver(),
        candidate_k=5,
        context_k=2,
    )
    result = search.search("Tell me about SendGrid", QueryContext())

    assert "sendgrid-failure" in {item.chunk.logical_id for item in result.admitted}
    assert result.dense == [payments]
    assert result.lexical[0].chunk.logical_id == "sendgrid-failure"


def test_hybrid_search_refresh_rebuilds_lexical_corpus() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failure.",
        logical_id="sendgrid-failure",
    )
    objects = FakeObjectStore()
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(),
        objects=objects,
        resolver=EvidenceResolver(),
    )
    assert search.search("SendGrid", QueryContext()).lexical == []

    key = "chunks/sendgrid-failure/2.0/chunk-000.json"
    objects.put_text(key, sendgrid.chunk.model_dump_json())
    search.refresh()

    assert search.search("SendGrid", QueryContext()).lexical[0].chunk.logical_id == "sendgrid-failure"
