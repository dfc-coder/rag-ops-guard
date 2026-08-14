from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_ops_guard.adapters.embeddings import llamacpp_embeddings
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm import llamacpp_chat, tokenizer
from rag_ops_guard.adapters.llm.llamacpp_chat import LlamaCppChatAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.domain.models import GroundedAnswer


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

    def invoke(self, prompt: object) -> object:
        del prompt
        fields = getattr(self.schema, "model_fields", {})
        if "normalized_question" in fields:
            return {
                "normalized_question": "normalized",
                "systems": ["payments"],
                "environment": "production",
                "api_version": None,
                "requires_clarification": False,
                "clarification_question": None,
                "safety_category": "normal",
                "fallback_message": "This request cannot be answered safely.",
            }
        return {
            "status": "answered",
            "answer": "grounded",
            "citation_ids": ["doc:1.0:000:deadbeef"],
        }


class FakeChat:
    instances: list[FakeChat] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.last_request: object | None = None
        self.instances.append(self)

    def with_structured_output(self, schema: type[object], method: str) -> FakeStructured:
        assert method == "json_schema"
        return FakeStructured(schema)

    def invoke(self, request: object) -> AIMessage:
        self.last_request = request
        return AIMessage(content="chat response")


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
    FakeChat.instances.clear()
    monkeypatch.setattr(llamacpp_chat, "ChatOpenAI", FakeChat)
    adapter = LlamaCppChatAdapter(
        "http://localhost:8080/v1",
        "qwen",
        0.7,
        analysis_max_tokens=128,
        answer_max_tokens=256,
        timeout_seconds=60.0,
        answer_system_prompt="grounding rules",
        chat_system_prompt="chat rules",
    )
    analysis = adapter.analyze_query("analyze")
    answer = adapter.generate_answer("answer")
    chat = adapter.generate_chat([HumanMessage(content="hello")])

    assert analysis.normalized_question == "normalized"
    assert analysis.insufficient_evidence_message == "This request cannot be answered safely."
    assert answer == GroundedAnswer(
        status="answered",
        answer="grounded",
        citation_ids=["doc:1.0:000:deadbeef"],
    )
    assert chat == "chat response"
    assert len(FakeChat.instances) == 2
    assert FakeChat.instances[0].kwargs["temperature"] == 0.0
    assert FakeChat.instances[0].kwargs["max_completion_tokens"] == 128
    assert FakeChat.instances[1].kwargs["max_completion_tokens"] == 256
    assert all(item.kwargs["timeout"] == 60.0 for item in FakeChat.instances)
    assert all(item.kwargs["max_retries"] == 0 for item in FakeChat.instances)
    request = FakeChat.instances[1].last_request
    assert isinstance(request, list)
    assert isinstance(request[0], SystemMessage)


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
