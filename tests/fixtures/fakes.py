from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from rag_ops_guard.domain.models import (
    Chunk,
    Evidence,
    GroundedAnswer,
    QueryAnalysis,
)


@dataclass
class FakeObjectStore:
    values: dict[str, str] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)

    def get_text(self, key: str) -> str:
        return self.values[key]

    def put_text(self, key: str, content: str, content_type: str = "text/plain") -> None:
        del content_type
        self.values[key] = content

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.values.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.values

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self.values if key.startswith(prefix))


@dataclass
class FakeEmbeddingProvider:
    dimension: int = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        value = float((sum(ord(char) for char in text) % 7) + 1)
        vector = [0.0] * self.dimension
        vector[0] = value
        return vector


@dataclass
class FakeReranker:
    default_score: float = 0.9
    scores_by_document: dict[str, float] = field(default_factory=dict)
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [
            next(
                (score for marker, score in self.scores_by_document.items() if marker in document),
                self.default_score,
            )
            for document in documents
        ]


@dataclass
class FakeVectorStore:
    evidence: list[Evidence] = field(default_factory=list)
    stored: dict[str, tuple[Chunk, list[float]]] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)

    def put(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.stored[chunk.id] = (chunk, embedding)
        return [chunk.id for chunk in chunks]

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[Evidence]:
        del embedding, filters
        return self.evidence[:top_k]

    def delete(self, keys: list[str]) -> None:
        self.deleted.extend(keys)
        for key in keys:
            self.stored.pop(key, None)


@dataclass
class FakeChatModel:
    analysis: QueryAnalysis
    answer: GroundedAnswer
    analysis_calls: int = 0
    generation_calls: int = 0
    rewrite_calls: int = 0

    def analyze_query(self, prompt: str) -> QueryAnalysis:
        del prompt
        self.analysis_calls += 1
        return self.analysis

    def rewrite_query(
        self,
        current_question: str,
        previous_query: str,
        source_titles: list[str],
    ) -> str:
        del source_titles
        self.rewrite_calls += 1
        return f"{previous_query} {current_question}".strip()

    def generate_answer(
        self,
        prompt: str,
        history: list[BaseMessage] | None = None,
    ) -> GroundedAnswer:
        del prompt, history
        self.generation_calls += 1
        return self.answer

    def generate_grounded_text(self, prompt: str) -> str:
        del prompt
        self.generation_calls += 1
        return self.answer.answer

    def generate_chat(self, messages: list[BaseMessage]) -> str:
        del messages
        self.generation_calls += 1
        return self.answer.answer
