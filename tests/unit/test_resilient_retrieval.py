from __future__ import annotations

from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports import RerankGrade
from rag_ops_guard.retrieval.hybrid import _explicit_query_anchors, _select_admitted_pairs
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
        text=(
            "Do not exceed three automated retries. "
            "If the third fails, alert Treasury Integrations."
        ),
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


def test_authority_breaks_only_near_relevance_ties_for_context_selection() -> None:
    vendor_meta = metadata(
        doc_id="vendor-v1",
        logical_id="vendor-note",
        authority=20,
    )
    vendor_meta.title = "Vendor Note"
    policy_meta = metadata(
        doc_id="policy-v2",
        logical_id="retry-policy",
        authority=100,
    )
    policy_meta.title = "Payment Retry Policy"
    api_meta = metadata(
        doc_id="api-v2",
        logical_id="payments-api",
        authority=95,
    )
    api_meta.title = "Payments API v2"
    incident_meta = metadata(
        doc_id="incident-v1",
        logical_id="incident",
        authority=70,
    )
    incident_meta.title = "Calypso Incident"

    vendor = evidence(meta=vendor_meta, text="Calypso retry supporting note")
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


def test_authority_does_not_rescue_irrelevant_evidence() -> None:
    low_meta = metadata(doc_id="relevant-v1", logical_id="relevant", authority=20)
    low_meta.title = "Strong Relevant Note"
    high_meta = metadata(doc_id="irrelevant-v1", logical_id="irrelevant", authority=100)
    high_meta.title = "High Authority Irrelevant"
    relevant = evidence(meta=low_meta)
    irrelevant = evidence(meta=high_meta)

    selected = _select_admitted_pairs(
        [
            (relevant, RerankGrade(relevant=True, score=0.8)),
            (irrelevant, RerankGrade(relevant=False, score=0.79)),
        ],
        limit=2,
    )

    assert [item.chunk.title for item, _grade in selected] == ["Strong Relevant Note"]


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


def test_unknown_lowercase_operational_target_is_inferred_as_anchor() -> None:
    anchors = _explicit_query_anchors("cuantos retries permite xarlatan?")
    assert "xarlatan" in anchors


def test_generic_subject_is_not_inferred_as_lowercase_target() -> None:
    anchors = _explicit_query_anchors("cuantos retries permite el sistema?")
    assert "sistema" not in anchors
    assert "el" not in anchors


def test_short_entity_first_query_still_keeps_entity_anchor() -> None:
    anchors = _explicit_query_anchors("Calypso retries?")
    assert "calypso" in anchors
