from __future__ import annotations

from typing import Protocol

from langchain_core.messages import BaseMessage

from rag_ops_guard.domain.models import Chunk, Evidence, GroundedAnswer, QueryAnalysis


class ChatModel(Protocol):
    def analyze_query(self, prompt: str) -> QueryAnalysis: ...

    def generate_answer(self, prompt: str) -> GroundedAnswer: ...

    def generate_chat(self, messages: list[BaseMessage]) -> str: ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def put(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[Evidence]: ...

    def delete(self, keys: list[str]) -> None: ...


class ObjectStore(Protocol):
    def get_text(self, key: str) -> str: ...

    def put_text(self, key: str, content: str, content_type: str = "text/plain") -> None: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str) -> list[str]: ...
