from __future__ import annotations

from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports import RerankGrade
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, _select_admitted_pairs
from rag_ops_guard.retrieval.resilient import ResilientKnowledgeSearch
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import (
    FakeEmbeddingProvider,
    FakeObjectStore,
    FakeReranker,
    FakeVectorStore,
)


class FailingKnowledgeSearch:
    def search(self, *args, **kwargs):
        raise RuntimeError("backend unavailable")

    def refresh(self) -> None:
        raise RuntimeError("backend unavailable")


class FakeFallbackKnowledgeSearch:
    def __init__(self) -> None:
        self.calls = 0
        self.refreshed = False

    def search(self, *args, **kwargs):
        self.calls += 1
        return "fallback-result"

    def refresh(self) -> None:
        self.refreshed = True


def test_resilient_search_falls_back_on_primary_failure() -> None:
    fallback = FakeFallbackKnowledgeSearch()
    search = ResilientKnowledgeSearch(FailingKnowledgeSearch(), fallback)

    result = search.search("query", QueryContext())

    assert result == "fallback-result"
    assert fallback.calls == 1


def test_resilient_refresh_updates_both_backends() -> None:
    primary = FakeFallbackKnowledgeSearch()
    fallback = FakeFallbackKnowledgeSearch()
    search = ResilientKnowledgeSearch(primary, fallback)

    search.refresh()

    assert primary.refreshed is True
    assert fallback.refreshed is True


def test_resilient_search_uses_primary_when_available() -> None:
    primary = FakeFallbackKnowledgeSearch()
    fallback = FakeFallbackKnowledgeSearch()
    search = ResilientKnowledgeSearch(primary, fallback)

    result = search.search("query", QueryContext())

    assert result == "fallback-result"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_reranker_prefers_relevance_before_authority() -> None:
    objects = FakeObjectStore()
    embeddings = FakeEmbeddingProvider()
    low_meta = metadata(doc_id="vendor-v1", logical_id="vendor", authority=10)
    low_meta.title = "Vendor Note"
    policy_meta = metadata(doc_id="policy-v2", logical_id="policy", authority=100)
    policy_meta.title = "Payment Retry Policy"
    api_meta = metadata(doc_id="api-v2", logical_id="api", authority=80)
    api_meta.title = "Payments API v2"
    incident_meta = metadata(doc_id="incident-v1", logical_id="incident", authority=60)
    incident_meta.title = "Calypso Incident"

    vendor = evidence(meta=low_meta, text="Calypso retry supporting note")
    policy = evidence(meta=policy_meta, text="Calypso retry canonical policy")
    api = evidence(meta=api_meta, text="Calypso retry API contract")
    incident = evidence(meta=incident_meta, text="Calypso retry incident evidence")
    ranked = [
        (vendor, RerankGrade(relevant=True, score=0.529)),
        (api, RerankGrade(relevant=True, score=0.515)),
        (incident, RerankGrade(relevant=True, score=0.506)),
        (policy, RerankGrade(relevant=True, score=0.501)),
    ]

    selected = _select_admitted_pairs(ranked, limit=3)
    titles = [item.chunk.title for item, _grade in selected]

    assert "Payment Retry Policy" in titles
    assert "Payments API v2" in titles
    assert "Vendor Note" not in titles


def test_authority_does_not_rescue_below_floor_evidence() -> None:
    low_meta = metadata(doc_id="relevant-v1", logical_id="relevant", authority=20)
    low_meta.title = "Strong Relevant Note"
    high_meta = metadata(doc_id="irrelevant-v1", logical_id="irrelevant", authority=100)
    high_meta.title = "High Authority Irrelevant"
    relevant = evidence(meta=low_meta)
    irrelevant = evidence(meta=high_meta)

    selected = _select_admitted_pairs(
        [
            (relevant, RerankGrade(relevant=True, score=0.8)),
            (irrelevant, RerankGrade(relevant=False, score=0.49)),
        ],
        limit=2,
        min_relevance=0.5,
    )

    assert [item.chunk.title for item, _grade in selected] == ["Strong Relevant Note"]


def test_authority_does_not_override_a_material_relevance_gap() -> None:
    low_meta = metadata(doc_id="strong-v1", logical_id="strong", authority=20)
    low_meta.title = "Strong Relevant Note"
    high_meta = metadata(doc_id="policy-v1", logical_id="policy", authority=100)
    high_meta.title = "Distant Policy"
    strong = evidence(meta=low_meta)
    policy = evidence(meta=high_meta)

    selected = _select_admitted_pairs(
        [
            (strong, RerankGrade(relevant=True, score=0.90)),
            (policy, RerankGrade(relevant=True, score=0.70)),
        ],
        limit=1,
    )

    assert [item.chunk.title for item, _grade in selected] == ["Strong Relevant Note"]


def test_knowledge_search_returns_unsupported_when_no_candidate_is_relevant() -> None:
    objects = FakeObjectStore()
    embeddings = FakeEmbeddingProvider()
    vectors = FakeVectorStore()
    item = evidence(text="unrelated internal note")
    vectors.results = [item]
    reranker = FakeReranker(
        grades=[RerankGrade(relevant=False, score=0.1)],
    )
    search = KnowledgeSearch(
        embeddings=embeddings,
        vectors=vectors,
        objects=objects,
        resolver=__import__(
            "rag_ops_guard.retrieval.resolver", fromlist=["EvidenceResolver"]
        ).EvidenceResolver(),
        reranker=reranker,
    )

    result = search.search("Write a Fibonacci function in Python.", QueryContext())

    assert result.supported is False
    assert result.admitted == []
    assert result.relevance == 0.1


def test_configured_relevance_floor_controls_admission() -> None:
    objects = FakeObjectStore()
    embeddings = FakeEmbeddingProvider()
    vectors = FakeVectorStore()
    item = evidence(text="borderline but related operational note")
    vectors.results = [item]
    reranker = FakeReranker(
        grades=[RerankGrade(relevant=True, score=0.62)],
    )
    search = KnowledgeSearch(
        embeddings=embeddings,
        vectors=vectors,
        objects=objects,
        resolver=__import__(
            "rag_ops_guard.retrieval.resolver", fromlist=["EvidenceResolver"]
        ).EvidenceResolver(),
        reranker=reranker,
        min_relevance=0.70,
    )

    result = search.search("operational note", QueryContext())

    assert result.supported is False
    assert result.admitted == []
    assert result.relevance == 0.62
