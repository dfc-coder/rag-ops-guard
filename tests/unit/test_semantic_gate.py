from __future__ import annotations

from rag_ops_guard.agent.semantic_gate import (
    GroundingAction,
    SemanticGateContext,
    SemanticGroundingGate,
)


class FixedEmbeddings:
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 3
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def embed_query(self, text: str) -> list[float]:
        assert "Current request:" in text
        return self.query_vector


CONTEXT = SemanticGateContext(
    has_grounded_context=True,
    has_active_evidence=True,
)


def test_gate_selects_best_semantic_action() -> None:
    gate = SemanticGroundingGate(
        FixedEmbeddings([0.05, 0.95, 0.0]),
        min_score=0.35,
        min_margin=0.015,
    )

    decision = gate.decide("current request", CONTEXT)

    assert decision.action == GroundingAction.RETRIEVE
    assert decision.score > 0.9
    assert decision.margin > 0.8


def test_gate_returns_uncertain_below_score_threshold() -> None:
    gate = SemanticGroundingGate(
        FixedEmbeddings([0.6, 0.5, 0.0]),
        min_score=0.8,
        min_margin=0.015,
    )

    decision = gate.decide("current request", CONTEXT)

    assert decision.score < 0.8
    assert decision.margin > 0.015
    assert decision.action == GroundingAction.UNCERTAIN


def test_gate_returns_uncertain_when_margin_is_too_small() -> None:
    gate = SemanticGroundingGate(
        FixedEmbeddings([0.71, 0.70, 0.0]),
        min_score=0.35,
        min_margin=0.05,
    )

    decision = gate.decide("current request", CONTEXT)

    assert decision.action == GroundingAction.UNCERTAIN


def test_gate_context_contains_state_flags_but_not_prior_subject_text() -> None:
    rendered = CONTEXT.render("Cuantos retries permite Xarlatan?")

    assert "Xarlatan" in rendered
    assert "Grounded conversational context available: True" in rendered
    assert "Active compatible evidence available: True" in rendered
    assert "Calypso" not in rendered
