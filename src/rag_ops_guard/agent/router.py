from __future__ import annotations

import math
from typing import Literal

from rag_ops_guard.ports import EmbeddingProvider

Route = Literal["chat", "knowledge"]

DEFAULT_CHAT_EXAMPLES = [
    "Hola, ¿qué haces?",
    "¿Quién eres y en qué puedes ayudarme?",
    "Gracias por la ayuda.",
    "Hello, what can you do?",
    "Who are you?",
]

DEFAULT_KNOWLEDGE_EXAMPLES = [
    "¿Qué dice la documentación sobre este sistema?",
    "Contame sobre este sistema o servicio.",
    "¿Qué sabemos de esta plataforma?",
    "¿Qué política aplica en este caso?",
    "¿Qué pasó en este incidente?",
    "¿Cuál es el SLA documentado?",
    "¿Qué runbook debo consultar?",
    "Tell me about this system or service.",
    "What does the operational documentation say?",
    "What happened in this incident?",
]


class SemanticRouter:
    """Embedding-based two-way router with no domain/entity-specific rules."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        chat_examples: list[str] | None = None,
        knowledge_examples: list[str] | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._chat = self._centroid(chat_examples or DEFAULT_CHAT_EXAMPLES)
        self._knowledge = self._centroid(knowledge_examples or DEFAULT_KNOWLEDGE_EXAMPLES)

    def route(self, text: str) -> Route:
        vector = self._embeddings.embed_query(text.strip())
        chat_score = _cosine(vector, self._chat)
        knowledge_score = _cosine(vector, self._knowledge)
        return "knowledge" if knowledge_score >= chat_score else "chat"

    def _centroid(self, examples: list[str]) -> list[float]:
        vectors = self._embeddings.embed_documents(examples)
        if not vectors:
            raise ValueError("semantic router requires at least one example")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("semantic router embeddings must share the same dimension")
        return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
