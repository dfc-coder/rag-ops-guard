from __future__ import annotations

from rag_ops_guard.domain.models import DocumentType, Evidence, QueryContext
from rag_ops_guard.retrieval.bm25 import BM25Index
from rag_ops_guard.retrieval.hybrid import (
    KnowledgeSearch,
    reciprocal_rank_fusion,
    retrieval_relevance,
)
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import (
    FakeEmbeddingProvider,
    FakeObjectStore,
    FakeReranker,
    FakeVectorStore,
)


def _named_evidence(title: str, text: str, *, logical_id: str, distance: float = 0.4) -> Evidence:
    meta = metadata(doc_id=f"{logical_id}-v1", logical_id=logical_id)
    meta.title = title
    meta.document_type = DocumentType.RUNBOOK
    item = evidence(meta=meta, text=text, distance=distance)
    item.chunk.title = title
    return item


def _store_documents(objects: FakeObjectStore, items: list[Evidence]) -> None:
    for item in items:
        key = (
            f"chunks/{item.chunk.logical_id}/{item.chunk.version}/"
            f"chunk-{item.chunk.chunk_index:03d}.json"
        )
        objects.put_text(key, item.chunk.model_dump_json())


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
    _store_documents(objects, [sendgrid, payments])

    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[payments]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=FakeReranker(),
        candidate_k=5,
        context_k=2,
    )
    result = search.search("Tell me about SendGrid", QueryContext())

    assert "sendgrid-failure" in {item.chunk.logical_id for item in result.admitted}
    assert result.dense == [payments]
    assert result.lexical[0].chunk.logical_id == "sendgrid-failure"
    assert result.supported is True


def test_probe_mode_embeds_raw_query_without_operational_instruction() -> None:
    class RecordingEmbeddings(FakeEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return super().embed_query(text)

    embeddings = RecordingEmbeddings()
    search = KnowledgeSearch(
        embeddings=embeddings,
        vectors=FakeVectorStore(),
        objects=FakeObjectStore(),
        resolver=EvidenceResolver(),
        reranker=FakeReranker(),
    )

    search.search("¿Qué haces?", QueryContext(), query_mode="probe")

    assert embeddings.queries == ["¿Qué haces?"]


def test_knowledge_mode_keeps_operational_embedding_instruction() -> None:
    class RecordingEmbeddings(FakeEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return super().embed_query(text)

    embeddings = RecordingEmbeddings()
    search = KnowledgeSearch(
        embeddings=embeddings,
        vectors=FakeVectorStore(),
        objects=FakeObjectStore(),
        resolver=EvidenceResolver(),
        reranker=FakeReranker(),
    )

    search.search("¿Qué pasa con SendGrid?", QueryContext(), query_mode="knowledge")

    assert "Instruct:" in embeddings.queries[0]
    assert "SendGrid" in embeddings.queries[0]


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
        reranker=FakeReranker(),
    )
    assert search.search("SendGrid", QueryContext()).lexical == []

    key = "chunks/sendgrid-failure/2.0/chunk-000.json"
    objects.put_text(key, sendgrid.chunk.model_dump_json())
    search.refresh()

    assert (
        search.search("SendGrid", QueryContext()).lexical[0].chunk.logical_id == "sendgrid-failure"
    )


def test_cross_encoder_admits_cross_language_retry_policy() -> None:
    calypso = _named_evidence(
        "Calypso Integration API",
        "The Calypso adapter accepts payment instructions from Payments API.",
        logical_id="calypso-api",
        distance=0.35,
    )
    retries = _named_evidence(
        "Payment Retry Policy",
        "Calypso timeout failures allow a maximum of three automated retries.",
        logical_id="payment-retry-policy",
        distance=0.45,
    )
    objects = FakeObjectStore()
    _store_documents(objects, [calypso, retries])
    reranker = FakeReranker(
        default_score=0.2,
        scores_by_document={
            "Payment Retry Policy": 0.94,
            "Calypso Integration API": 0.58,
        },
    )
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[calypso, retries]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=reranker,
        candidate_k=5,
        context_k=4,
        min_reranker_score=0.5,
    )

    result = search.search(
        "Cuantos reintentos permite Calypso?", QueryContext(), query_mode="probe"
    )

    assert result.supported is True
    assert result.relevance == 0.94
    assert result.admitted[0].chunk.title == "Payment Retry Policy"
    assert "reintentos" in reranker.calls[0][0].casefold()


def test_cross_encoder_rejects_candidates_below_support_threshold() -> None:
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.1,
    )
    objects = FakeObjectStore()
    _store_documents(objects, [payments])
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[payments]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=FakeReranker(default_score=0.1),
        min_reranker_score=0.5,
    )

    result = search.search("Cual es la capital de Francia?", QueryContext(), query_mode="probe")

    assert result.supported is False
    assert result.relevance == 0.0
    assert result.admitted == []


def test_relevance_accepts_strong_admitted_support() -> None:
    calypso = _named_evidence(
        "Calypso Integration API",
        "The Calypso adapter accepts payment instructions from Payments API.",
        logical_id="calypso-api",
        distance=0.3,
    )

    score = retrieval_relevance(
        "¿Cuál es el objetivo de Calypso Payments API?",
        admitted=[calypso],
    )

    assert score >= 0.4


def test_relevance_rejects_weak_admitted_result_without_lexical_support() -> None:
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.8,
    )

    score = retrieval_relevance(
        "¿Cuál es la capital de Francia?",
        admitted=[payments],
    )

    assert score < 0.4


def test_rejected_dense_candidate_cannot_raise_final_relevance() -> None:
    rejected = _named_evidence(
        "France Geography",
        "Paris is the capital of France.",
        logical_id="rejected",
        distance=0.01,
    )
    admitted = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.8,
    )

    without_rejected = retrieval_relevance("capital de Francia", admitted=[admitted])
    with_only_admitted = retrieval_relevance("capital de Francia", admitted=[admitted])

    assert rejected.distance == 0.01
    assert with_only_admitted == without_rejected
    assert with_only_admitted < 0.4
