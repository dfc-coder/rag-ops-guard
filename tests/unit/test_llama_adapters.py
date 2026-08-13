from __future__ import annotations

from typing import Any

import pytest

from rag_ops_guard.adapters.embeddings import llamacpp_embeddings
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm import llamacpp_chat, tokenizer
from rag_ops_guard.adapters.llm.llamacpp_chat import LlamaCppChatAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.domain.models import GroundedAnswer, QueryAnalysis


class FakeEmbeddings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [3.0, 4.0]


class FakeStructured:
    def __init__(self, schema: type[object]) -> None:
        self.schema = schema

    def invoke(self, prompt: str) -> object:
        del prompt
        if issubclass(self.schema, QueryAnalysis):
            return {
                "normalized_question": "normalized",
                "systems": ["payments"],
                "environment": "production",
                "api_version": None,
                "requires_clarification": False,
                "clarification_question": None,
                "safety_category": "normal",
                "safety_blocked_message": "This request cannot be fulfilled safely.",
                "insufficient_evidence_message": "There is not enough evidence to answer safely.",
            }
        return {
            "status": "answered",
            "answer": "grounded",
            "citation_ids": ["doc:1.0:000:deadbeef"],
        }


class FakeChat:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def with_structured_output(self, schema: type[object], method: str) -> FakeStructured:
        assert method == "json_schema"
        return FakeStructured(schema)


class FakeHttpResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"tokens": [1, 2, 3]}


def test_embedding_adapter_normalizes_and_validates_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", FakeEmbeddings)
    adapter = LlamaCppEmbeddingAdapter("http://localhost:8081/v1", "embed", dimension=2)
    assert adapter.embed_query("hello") == pytest.approx([0.6, 0.8])
    assert adapter.embed_documents(["a", "b"])[1] == pytest.approx([0.6, 0.8])


def test_chat_adapter_uses_structured_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp_chat, "ChatOpenAI", FakeChat)
    adapter = LlamaCppChatAdapter("http://localhost:8080/v1", "qwen", 0.0, 128)
    analysis = adapter.analyze_query("analyze")
    answer = adapter.generate_answer("answer")
    assert analysis.normalized_question == "normalized"
    assert analysis.insufficient_evidence_message == "There is not enough evidence to answer safely."
    assert answer == GroundedAnswer(
        status="answered",
        answer="grounded",
        citation_ids=["doc:1.0:000:deadbeef"],
    )


def test_token_counter_calls_llama_tokenize(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeHttpResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeHttpResponse()

    monkeypatch.setattr(tokenizer.httpx, "post", fake_post)
    counter = LlamaCppTokenCounter("http://localhost:8081/v1")
    assert counter("hello") == 3
    assert captured["url"] == "http://localhost:8081/tokenize"
