from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from rag_ops_guard.ports import EmbeddingProvider

Route = Literal[
    "chat",
    "capabilities",
    "catalog",
    "knowledge",
    "out_of_scope",
    "uncertain",
]

DEFAULT_ROUTE_EXAMPLES: dict[Route, list[str]] = {
    "chat": [
        "Hola",
        "Gracias por la ayuda.",
        "Buen día",
        "Hello",
        "Thanks for your help.",
    ],
    "capabilities": [
        "¿Qué puedes hacer?",
        "¿Cómo puedes ayudarme?",
        "¿Quién eres y para qué sirves?",
        "What can you do?",
        "How can you help me?",
    ],
    "catalog": [
        "¿Qué documentación tienes disponible?",
        "¿Qué documentos puedo consultar?",
        "Mostrame qué runbooks y APIs están disponibles.",
        "What documentation is available?",
        "Which documents can I consult?",
    ],
    "knowledge": [
        "¿Qué dice la documentación sobre este sistema?",
        "Contame sobre este sistema o servicio.",
        "¿Qué política aplica en este caso?",
        "¿Qué pasó en este incidente?",
        "¿Cuál es el SLA documentado?",
        "¿Qué runbook debo consultar?",
        "¿Qué hace esta API?",
        "¿Qué ocurre si falla esta integración?",
        "Tell me about this system or service.",
        "What does the operational documentation say?",
    ],
    "out_of_scope": [
        "¿Cuál es la capital de Francia?",
        "¿Qué temperatura hace hoy?",
        "Dame una receta de pizza.",
        "Who won the football match?",
        "What is the weather today?",
    ],
    "uncertain": [],
}


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    score: float
    margin: float
    scores: dict[str, float]


class SemanticRouter:
    """Embedding router with per-example scoring, confidence and abstention."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        route_examples: dict[Route, list[str]] | None = None,
        min_score: float = 0.35,
        min_margin: float = 0.015,
    ) -> None:
        self._embeddings = embeddings
        self._min_score = min_score
        self._min_margin = min_margin
        examples = route_examples or DEFAULT_ROUTE_EXAMPLES
        self._vectors: dict[Route, list[list[float]]] = {}
        for route, utterances in examples.items():
            if route == "uncertain" or not utterances:
                continue
            vectors = embeddings.embed_documents(utterances)
            if not vectors:
                raise ValueError(f"semantic router requires examples for route {route}")
            self._vectors[route] = vectors

    def route(self, text: str) -> RouteDecision:
        vector = self._embeddings.embed_query(text.strip())
        scores = {
            route: max(_cosine(vector, example) for example in examples)
            for route, examples in self._vectors.items()
        }
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ordered:
            return RouteDecision(route="uncertain", score=0.0, margin=0.0, scores={})

        best_route, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_score - second_score
        route: Route = best_route
        if best_score < self._min_score or margin < self._min_margin:
            route = "uncertain"
        return RouteDecision(
            route=route,
            score=best_score,
            margin=margin,
            scores={key: round(value, 6) for key, value in scores.items()},
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
