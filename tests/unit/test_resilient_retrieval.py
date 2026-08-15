from __future__ import annotations

from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.retrieval.hybrid import _explicit_query_anchors
from rag_ops_guard.retrieval.resilient import ResilientKnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeReranker, FakeVectorStore


class ModeSensitiveEmbeddings(FakeEmbeddingProvider):
    def embed_query(self, text: str) -> list[float]:
        marker = 1.0 if text.startswith("Instruct:") else 2.0
        return [marker, 0.0, 0.0, 0.0]


class ModeSensitiveVectors(FakeVectorStore):
    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ):
        del filters
        if embedding[0] == 1.0:
            return []
        return self.evidence[:top_k]


def test_knowledge_search_retries_raw_query_without_lowering_admission_rules() -> None:
    meta = metadata(
        doc_id="calypso-timeout-runbook-v2",
        logical_id="calypso-timeout-runbook",
        system="payments",
        environment="production",
    )
    meta.title = "Calypso Timeout Runbook"
    item = evidence(
        meta=meta,
        text="Do not exceed three automated retries. If the third fails, alert Treasury Integrations.",
        distance=0.2,
    )
    item.chunk.title = "Calypso Timeout Runbook"

    search = ResilientKnowledgeSearch(
        embeddings=ModeSensitiveEmbeddings(),
        vectors=ModeSensitiveVectors(evidence=[item]),
        objects=FakeObjectStore(),
        resolver=EvidenceResolver(),
        reranker=FakeReranker(default_score=0.95, default_relevant=True),
        candidate_k=8,
        context_k=3,
    )

    result = search.search("Cuantos reintentos permite Calypso?", QueryContext())

    assert result.supported is True
    assert result.admitted
    assert result.admitted[0].chunk.logical_id == "calypso-timeout-runbook"
    assert result.relevance == 0.95


def test_contextual_query_only_requires_real_entity_anchor() -> None:
    anchors = _explicit_query_anchors(
        "Cuantos reintentos permite Calypso. Y despues del tercero?"
    )
    assert anchors == {"calypso"}


def test_synthetic_followup_label_cannot_become_required_anchor() -> None:
    anchors = _explicit_query_anchors(
        "Cuantos reintentos permite Calypso. Follow-up: Y despues del tercero?"
    )
    assert "follow" not in anchors
    assert "up" not in anchors
    assert "calypso" in anchors


def test_sentence_initial_temporal_word_is_not_treated_as_entity() -> None:
    assert _explicit_query_anchors("Despues del tercero, que pasa?") == set()


def test_unknown_named_operational_target_remains_fail_closed_anchor() -> None:
    anchors = _explicit_query_anchors("Cuantos retries permite Xarlatan?")
    assert "xarlatan" in anchors


def test_short_entity_first_query_still_keeps_entity_anchor() -> None:
    anchors = _explicit_query_anchors("Calypso retries?")
    assert "calypso" in anchors
