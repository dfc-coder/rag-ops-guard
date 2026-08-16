from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from rag_ops_guard.domain.models import Chunk, Evidence

T = TypeVar("T")
MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class RerankGrade:
    """Learned relevance decision for one query-document pair."""

    relevant: bool
    score: float


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelMessage:
    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ModelTurn:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


class ToolResult(BaseModel):
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def schema(self) -> dict[str, Any]: ...

    def invoke(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolCallingModel(Protocol):
    def bind_tools(self, tools: list[Tool]) -> Self: ...

    def invoke(self, messages: list[ModelMessage]) -> ModelTurn: ...

    def invoke_structured(self, messages: list[ModelMessage], schema: type[T]) -> T: ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    """Grade query-document pairs with a learned relevance model."""

    def grade(self, query: str, documents: list[str]) -> list[RerankGrade]: ...


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

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str) -> list[str]: ...
