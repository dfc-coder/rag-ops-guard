from __future__ import annotations

from rag_ops_guard.domain.models import DocumentType, Evidence, QueryContext
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch
from rag_ops_guard.retrieval.identifiers import explicit_identifiers, focus_allows_rewrite
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import (
    FakeEmbeddingProvider,
    FakeObjectStore,
    FakeReranker,
    FakeVectorStore,
)


def _named_evidence(title: str, text: str, *, logical_id: str) -> Evidence:
    meta = metadata(doc_id=f"{logical_id}-v1", logical_id=logical_id)
    meta.title = title
    meta.document_type = DocumentType.RUNBOOK
    item = evidence(meta=meta, text=text, distance=0.1)
    item.chunk.title = title
    return item


def _store(objects: FakeObjectStore, item: Evidence) -> None:
    key = (
        f"chunks/{item.chunk.logical_id}/{item.chunk.version}/"
        f"chunk-{item.chunk.chunk_index:03d}.json"
    )
    objects.put_text(key, item.chunk.model_dump_json())


def test_explicit_identifier_extraction_is_generic_not_entity_allowlisted() -> None:
    assert explicit_identifiers("Cual es el timeout exacto de SAP en produccion?") == {"sap"}
    assert "sendgrid" in explicit_identifiers("Que pasa con SendGrid?")
    assert "calypso" in explicit_identifiers("Cuantos reintentos permite Calypso?")
    assert explicit_identifiers("Y despues del tercero?") == set()


def test_trusted_focus_allows_elliptical_followup_but_blocks_named_topic_switch() -> None:
    previous = "Cuantos reintentos permite Calypso?"
    titles = ["Payment Retry Policy"]

    assert focus_allows_rewrite("Y despues del tercero?", previous, titles) is True
    assert (
        focus_allows_rewrite("Cual es el timeout exacto de SAP en produccion?", previous, titles)
        is False
    )


def test_explicit_unknown_identifier_rejects_high_scoring_domain_near_miss() -> None:
    calypso = _named_evidence(
        "Calypso Timeout Runbook",
        "Production Calypso timeout handling and escalation guidance.",
        logical_id="calypso-timeout-runbook",
    )
    objects = FakeObjectStore()
    _store(objects, calypso)
    reranker = FakeReranker(default_score=0.99)
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[calypso]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=reranker,
        candidate_k=5,
        context_k=4,
        min_reranker_score=0.1,
    )

    result = search.search(
        "Cual es el timeout exacto de SAP en produccion?",
        QueryContext(),
        query_mode="probe",
    )

    assert result.supported is False
    assert result.admitted == []
    assert result.relevance == 0.0
    assert reranker.calls[0][1] == []


def test_known_identifier_keeps_matching_cross_language_candidate() -> None:
    retries = _named_evidence(
        "Payment Retry Policy",
        "Transient Calypso timeouts allow a maximum of three automated retries.",
        logical_id="payment-retry-policy",
    )
    objects = FakeObjectStore()
    _store(objects, retries)
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[retries]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=FakeReranker(default_score=0.9),
        candidate_k=5,
        context_k=4,
        min_reranker_score=0.1,
    )

    result = search.search(
        "Cuantos reintentos permite Calypso?",
        QueryContext(),
        query_mode="probe",
    )

    assert result.supported is True
    assert result.admitted[0].chunk.title == "Payment Retry Policy"
