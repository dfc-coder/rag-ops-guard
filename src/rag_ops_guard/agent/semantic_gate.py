from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter
from rag_ops_guard.config import get_settings
from rag_ops_guard.ports import Reranker


class GroundingAction(StrEnum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"
    CATALOG = "catalog"
    UNCERTAIN = "uncertain"


POLICY_HYPOTHESES: dict[GroundingAction, str] = {
    GroundingAction.DIRECT: (
        "This turn can be completed without obtaining a new private or internal operational fact. "
        "It is ordinary conversation, general knowledge, self-contained coding, or a "
        "transformation of information already visible in the conversation."
    ),
    GroundingAction.RETRIEVE: (
        "This turn requires a new private or internal operational fact, verification, policy, "
        "configuration, incident detail, system behavior, or factual continuation of a previously "
        "grounded operational topic."
    ),
    GroundingAction.CATALOG: (
        "This turn asks which private or internal documentation, runbooks, APIs, policies, "
        "incidents, or other knowledge sources are available to inspect."
    ),
}


@dataclass(frozen=True)
class SemanticGateContext:
    has_grounded_context: bool
    has_active_evidence: bool

    def render(self, message: str) -> str:
        grounded = "yes" if self.has_grounded_context else "no"
        active = "yes" if self.has_active_evidence else "no"
        return (
            "Classify the control action required by the current user turn.\n"
            f"Current user turn: {message.strip()}\n"
            f"A previously grounded internal topic exists: {grounded}.\n"
            f"Compatible prior evidence is active: {active}."
        )


@dataclass(frozen=True)
class GateDecision:
    action: GroundingAction
    score: float
    margin: float
    scores: dict[str, float]


class TurnGate(Protocol):
    def decide(self, message: str, context: SemanticGateContext) -> GateDecision: ...


class SemanticGroundingGate:
    """Cross-encoder semantic policy gate using the existing local reranker."""

    def __init__(
        self,
        reranker: Reranker,
        *,
        min_score: float,
        min_margin: float,
    ) -> None:
        self._reranker = reranker
        self._min_score = min_score
        self._min_margin = min_margin
        self._actions = list(POLICY_HYPOTHESES)
        self._hypotheses = [POLICY_HYPOTHESES[action] for action in self._actions]

    @classmethod
    def from_settings(cls) -> SemanticGroundingGate:
        settings = get_settings()
        return cls(
            LlamaCppRerankerAdapter(
                settings.reranker_base_url,
                settings.reranker_model,
                settings.reranker_timeout_seconds,
            ),
            min_score=settings.router_min_score,
            min_margin=settings.router_min_margin,
        )

    def decide(self, message: str, context: SemanticGateContext) -> GateDecision:
        grades = self._reranker.grade(context.render(message), self._hypotheses)
        if len(grades) != len(self._actions):
            raise ValueError("semantic gate reranker returned an unexpected grade count")

        scores = {
            action: grade.score
            for action, grade in zip(self._actions, grades, strict=True)
        }
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        if not ordered:
            return GateDecision(
                action=GroundingAction.UNCERTAIN,
                score=0.0,
                margin=0.0,
                scores={},
            )

        best_action, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_score - second_score
        action = (
            GroundingAction.UNCERTAIN
            if best_score < self._min_score or margin < self._min_margin
            else best_action
        )
        return GateDecision(
            action=action,
            score=best_score,
            margin=margin,
            scores={key.value: round(value, 6) for key, value in scores.items()},
        )
