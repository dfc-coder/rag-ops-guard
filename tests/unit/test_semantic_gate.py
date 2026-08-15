from __future__ import annotations

from rag_ops_guard.agent.semantic_gate import (
    GroundingAction,
    SemanticGateContext,
    SemanticGroundingGate,
)
from rag_ops_guard.ports import RerankGrade


class FixedReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def grade(self, query: str, documents: list[str]) -> list[RerankGrade]:
        self.calls.append((query, documents))
        return [RerankGrade(relevant=score >= 0.5, score=score) for score in self.scores]


CONTEXT = SemanticGateContext(
    has_grounded_context=True,
    has_active_evidence=True,
)


def test_gate_selects_best_cross_encoder_action() -> None:
    reranker = FixedReranker([0.1, 0.91, 0.05])
    gate = SemanticGroundingGate(reranker, min_score=0.35, min_margin=0.015)

    decision = gate.decide("Y si falla?", CONTEXT)

    assert decision.action == GroundingAction.RETRIEVE
    assert decision.score == 0.91
    assert decision.margin == 0.81
    assert len(reranker.calls[0][1]) == 3


def test_gate_returns_uncertain_below_score_threshold() -> None:
    gate = SemanticGroundingGate(
        FixedReranker([0.31, 0.30, 0.10]),
        min_score=0.35,
        min_margin=0.005,
    )

    decision = gate.decide("current request", CONTEXT)

    assert decision.action == GroundingAction.UNCERTAIN


def test_gate_returns_uncertain_when_margin_is_too_small() -> None:
    gate = SemanticGroundingGate(
        FixedReranker([0.72, 0.71, 0.10]),
        min_score=0.35,
        min_margin=0.05,
    )

    decision = gate.decide("current request", CONTEXT)

    assert decision.action == GroundingAction.UNCERTAIN


def test_gate_context_contains_state_flags_but_not_prior_subject_text() -> None:
    reranker = FixedReranker([0.10, 0.90, 0.05])
    gate = SemanticGroundingGate(reranker, min_score=0.35, min_margin=0.015)

    gate.decide("Cuantos retries permite Xarlatan?", CONTEXT)
    rendered = reranker.calls[0][0]

    assert "Xarlatan" in rendered
    assert "grounded internal topic exists: yes" in rendered
    assert "prior evidence is active: yes" in rendered
    assert "Calypso" not in rendered
