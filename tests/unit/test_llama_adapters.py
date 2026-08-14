from __future__ import annotations

from typing import Any, ClassVar

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
        self.last_request: object | None = None

    def invoke(self, prompt: object) -> object:
        self.last_request = prompt
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
        if set(fields) == {"answer"}:
            return {"answer": "grounded draft"}
        return {
            "status": "answered",
            "answer": "grounded",
            "citation_ids": ["doc:1.0:000:deadbeef"],
        }


class FakeChat:
    instances: ClassVar[list[FakeChat]] = []
    rewrite_response: ClassVar[str] = "Calypso retries after the third attempt"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.last_request: object | None = None
        self.requests: list[object] = []
        self.structured: list[FakeStructured] = []
        self.instances.append(self)

    def with_structured_output(self, schema: type[object], method: str) -> FakeStructured:
        assert method == "json_schema"
        structured = FakeStructured(schema)
        self.structured.append(structured)
        return structured

    def invoke(self, request: object) -> AIMessage:
        self.last_request = request
        self.requests.append(request)
        if (
            isinstance(request, list)
            and request
            and isinstance(request[0], SystemMessage)
            and "Rewrite CURRENT_QUESTION" in str(request[0].content)
        ):
            return AIMessage(content=self.rewrite_response)
        return AIMessage(content="chat response")


class FakeHttpResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"tokens": [1, 2, 3]}


def _chat_adapter(monkeypatch: pytest.MonkeyPatch) -> LlamaCppChatAdapter:
    FakeChat.instances.clear()
    FakeChat.rewrite_response = "Calypso retries after the third attempt"
    monkeypatch.setattr(llamacpp_chat, "ChatOpenAI", FakeChat)
    return LlamaCppChatAdapter(
        "http://localhost:8080/v1",
        "qwen",
        0.7,
        analysis_max_tokens=128,
        answer_max_tokens=256,
        timeout_seconds=60.0,
        answer_system_prompt="grounding rules",
        chat_system_prompt="chat rules",
    )


def test_embedding_adapter_normalizes_and_validates_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", FakeEmbeddings)
    adapter = LlamaCppEmbeddingAdapter("http://localhost:8081/v1", "embed", dimension=2)
    assert adapter.embed_query("hello") == pytest.approx([0.6, 0.8])
    assert adapter.embed_documents(["a", "b"])[1] == pytest.approx([0.6, 0.8])


def test_chat_adapter_uses_plain_contextual_rewrite_and_structured_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _chat_adapter(monkeypatch)
    analysis = adapter.analyze_query("analyze")
    rewrite = adapter.rewrite_query(
        current_question="¿Y después del tercero?",
        previous_query="reintentos de Calypso",
        source_titles=["Payment Retry Policy"],
    )
    rewrite_request = FakeChat.instances[0].requests[-1]
    answer = adapter.generate_answer("answer")
    grounded_text = adapter.generate_grounded_text("grounded answer")
    chat = adapter.generate_chat([HumanMessage(content="hello")])

    assert analysis.normalized_question == "normalized"
    assert analysis.insufficient_evidence_message == "This request cannot be answered safely."
    assert rewrite == "Calypso retries after the third attempt"
    assert answer == GroundedAnswer(
        status="answered",
        answer="grounded",
        citation_ids=["doc:1.0:000:deadbeef"],
    )
    assert grounded_text == "grounded draft"
    assert chat == "chat response"
    assert len(FakeChat.instances) == 2
    assert FakeChat.instances[0].kwargs["temperature"] == 0.0
    assert FakeChat.instances[0].kwargs["max_completion_tokens"] == 128
    assert FakeChat.instances[1].kwargs["temperature"] == 0.7
    assert FakeChat.instances[1].kwargs["max_completion_tokens"] == 256
    assert all(item.kwargs["timeout"] == 60.0 for item in FakeChat.instances)
    assert all(item.kwargs["max_retries"] == 0 for item in FakeChat.instances)
    assert isinstance(rewrite_request, list)
    assert isinstance(rewrite_request[0], SystemMessage)
    assert isinstance(rewrite_request[1], HumanMessage)

    grounded_request = FakeChat.instances[1].structured[1].last_request
    assert isinstance(grounded_request, list)
    assert isinstance(grounded_request[0], SystemMessage)
    assert isinstance(grounded_request[1], HumanMessage)


def test_self_contained_topic_switch_is_decided_by_contextualizer_not_name_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _chat_adapter(monkeypatch)
    question = "Cual es el timeout exacto de SAP en produccion?"
    FakeChat.rewrite_response = question

    rewritten = adapter.rewrite_query(
        current_question=question,
        previous_query="Cuantos reintentos permite Calypso?",
        source_titles=["Payment Retry Policy"],
    )

    assert rewritten == question
    assert len(FakeChat.instances[0].requests) == 1


def test_empty_contextualizer_output_falls_back_to_current_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _chat_adapter(monkeypatch)
    question = "Cual es el timeout exacto de SAP en produccion?"
    FakeChat.rewrite_response = ""

    rewritten = adapter.rewrite_query(
        current_question=question,
        previous_query="Cuantos reintentos permite Calypso?",
        source_titles=["Payment Retry Policy"],
    )

    assert rewritten == question


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
