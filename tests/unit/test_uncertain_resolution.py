from __future__ import annotations

from rag_ops_guard.graph.conversational_agent import _uncertain_prefers_knowledge
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult
from tests.fixtures.builders import evidence


def _result(*, relevance: float, lexical_relevance: float) -> KnowledgeSearchResult:
    item = evidence(text="Operational evidence")
    return KnowledgeSearchResult(
        dense=[item],
        lexical=[],
        fused=[item],
        admitted=[item],
        relevance=relevance,
        lexical_relevance=lexical_relevance,
    )


def test_dense_only_probe_does_not_override_leading_control_intent() -> None:
    scores = {
        "catalog": 0.624446,
        "knowledge": 0.618493,
        "out_of_scope": 0.595109,
        "capabilities": 0.579848,
        "chat": 0.579213,
    }

    assert not _uncertain_prefers_knowledge(
        scores,
        _result(relevance=0.472985, lexical_relevance=0.0),
        relevance_threshold=0.4,
    )


def test_admitted_lexical_anchor_can_confirm_knowledge_when_router_is_uncertain() -> None:
    scores = {
        "out_of_scope": 0.63,
        "knowledge": 0.625,
        "capabilities": 0.2,
        "chat": 0.1,
    }

    assert _uncertain_prefers_knowledge(
        scores,
        _result(relevance=0.75, lexical_relevance=1.0),
        relevance_threshold=0.4,
    )


def test_semantic_knowledge_lead_can_use_dense_evidence_without_lexical_anchor() -> None:
    scores = {
        "knowledge": 0.63,
        "out_of_scope": 0.625,
        "capabilities": 0.2,
        "chat": 0.1,
    }

    assert _uncertain_prefers_knowledge(
        scores,
        _result(relevance=0.6, lexical_relevance=0.0),
        relevance_threshold=0.4,
    )
