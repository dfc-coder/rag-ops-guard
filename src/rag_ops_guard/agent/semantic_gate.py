from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.config import get_settings
from rag_ops_guard.ports import EmbeddingProvider


class GroundingAction(StrEnum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"
    CATALOG = "catalog"
    UNCERTAIN = "uncertain"


ROUTE_SEMANTICS: dict[GroundingAction, str] = {
    GroundingAction.DIRECT: (
        "Handle the current request without new private or internal operational evidence. "
        "This includes general knowledge, ordinary conversation, self-contained coding, and "
        "transforming information already visible in the conversation."
    ),
    GroundingAction.RETRIEVE: (
        "The current request needs a new private or internal operational fact, verification, "
        "policy, configuration, incident detail, system behavior, or a factual follow-up that "
        "depends on previously grounded operational context."
    ),
    GroundingAction.CATALOG: (
        "The current request asks which internal documents, runbooks, APIs, policies, incidents, "
        "or other knowledge sources are available to inspect."
    ),
}


@dataclass(frozen=True)
class SemanticGateContext:
    has_grounded_context: bool
    has_active_evidence: bool

    def render(self, message: str) -> str:
        return (
            f"Current request:\n{message.strip()}\n\n"
            f"Grounded conversational context available: {self.has_grounded_context}\n"
            f"Active compatible evidence available: {self.has_active_evidence}"
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
    """Fast embedding router for the small semantic control surface."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        min_score: float,
        min_margin: float,
    ) -> None:
        self._embeddings = embeddings
        self._min_score = min_score
        self._min_margin = min_margin
        actions = list(ROUTE_SEMANTICS)
        vectors = embeddings.embed_documents([ROUTE_SEMANTICS[action] for action in actions])
        if len(vectors) != len(actions):
            raise ValueError("semantic gate did not receive one vector per action")
        self._vectors = dict(zip(actions, vectors, strict=True))

    @classmethod
    def from_settings(cls) -> SemanticGroundingGate:
        settings = get_settings()
        return cls(
            LlamaCppEmbeddingAdapter(
                settings.embedding_base_url,
                settings.embedding_model,
                settings.embedding_dimension,
            ),
            min_score=settings.router_min_score,
            min_margin=settings.router_min_margin,
        )

    def decide(self, message: str, context: SemanticGateContext) -> GateDecision:
        vector = self._embeddings.embed_query(context.render(message))
        scores = {
            action: _cosine(vector, prototype)
            for action, prototype in self._vectors.items()
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


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
