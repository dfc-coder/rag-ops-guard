from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_ops_guard.adapters.llm.openai_tool_calling import (
    OpenAIToolCallingAdapter,
    _content_text,
    _to_langchain_messages,
)
from rag_ops_guard.domain.models import StructuredAnswer
from rag_ops_guard.ports.interfaces import ModelMessage, ToolCall


def test_framework_adapter_maps_core_messages_to_langchain_messages() -> None:
    messages = [
        ModelMessage(role="system", content="system"),
        ModelMessage(role="user", content="question"),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="call-1", name="search_documents", arguments={"query": "x"}),),
        ),
        ModelMessage(
            role="tool",
            name="search_documents",
            tool_call_id="call-1",
            content='{"ok": true}',
        ),
    ]

    converted = _to_langchain_messages(messages)

    assert isinstance(converted[0], SystemMessage)
    assert converted[0].content == "system"
    assert isinstance(converted[1], HumanMessage)
    assert converted[1].content == "question"
    assert isinstance(converted[2], AIMessage)
    assert converted[2].tool_calls[0]["name"] == "search_documents"
    assert converted[2].tool_calls[0]["args"] == {"query": "x"}
    assert isinstance(converted[3], ToolMessage)
    assert converted[3].tool_call_id == "call-1"


def test_framework_adapter_rejects_tool_message_without_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        _to_langchain_messages([ModelMessage(role="tool", content="result")])


def test_content_text_normalizes_supported_openai_content_shapes() -> None:
    assert _content_text("plain") == "plain"
    assert _content_text([{"text": "hello"}, {"text": " world"}]) == "hello world"
    assert _content_text(["a", "b"]) == "ab"
    assert _content_text({"text": "ignored"}) == ""


class _FakeRunnable:
    def __init__(self, response: AIMessage) -> None:
        self._response = response

    def invoke(self, _messages: Any) -> AIMessage:
        return self._response


class _FakeModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.definitions: list[dict[str, Any]] | None = None
        self.tool_choice: Any = None
        self.parallel_tool_calls: bool | None = None
        self.response_format: dict[str, Any] | None = None
        self.bind_kwargs: dict[str, Any] = {}

    def bind_tools(
        self,
        definitions: list[dict[str, Any]],
        *,
        tool_choice: Any,
        parallel_tool_calls: bool,
    ) -> _FakeRunnable:
        self.definitions = definitions
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        return _FakeRunnable(self.response)

    def bind(self, **kwargs: Any) -> _FakeRunnable:
        self.bind_kwargs = dict(kwargs)
        response_format = kwargs.get("response_format")
        self.response_format = response_format if isinstance(response_format, dict) else None
        return _FakeRunnable(self.response)


def _adapter(max_completion_tokens: int = 512) -> OpenAIToolCallingAdapter:
    return OpenAIToolCallingAdapter(
        base_url="http://localhost:8080/v1",
        model="test",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=60,
    )


def test_structured_response_uses_schema_constrained_json_and_pydantic_validation() -> None:
    adapter = _adapter()
    fake = _FakeModel(
        AIMessage(
            content=(
                '{"segments":[{"text":"Three retries are allowed.",'
                '"citation_ids":["chunk-1"]}]}'
            )
        )
    )
    adapter._model = fake  # type: ignore[assignment]
    messages = [
        ModelMessage(role="system", content="system"),
        ModelMessage(role="user", content="x"),
    ]

    result = adapter.invoke_structured(messages, StructuredAnswer)

    assert result.segments[0].text == "Three retries are allowed."
    assert result.segments[0].citation_ids == ["chunk-1"]
    assert fake.response_format is not None
    schema = fake.response_format["schema"]
    citation_schema = schema["$defs"]["GeneratedSegment"]["properties"]["citation_ids"]
    assert fake.response_format["type"] == "json_object"
    assert citation_schema["maxItems"] == 0
    assert fake.bind_kwargs["temperature"] == 0.0
    assert fake.bind_kwargs["presence_penalty"] == 0.0
    assert fake.bind_kwargs["max_completion_tokens"] == 512


def test_structured_response_uses_configured_completion_budget_without_hidden_cap() -> None:
    adapter = _adapter(max_completion_tokens=768)
    fake = _FakeModel(
        AIMessage(content='{"segments":[{"text":"ok","citation_ids":[]}]}')
    )
    adapter._model = fake  # type: ignore[assignment]

    adapter.invoke_structured([ModelMessage(role="user", content="x")], StructuredAnswer)

    assert fake.bind_kwargs["max_completion_tokens"] == 768


def test_structured_response_limits_citation_ids_to_retrieved_sources() -> None:
    adapter = _adapter()
    fake = _FakeModel(
        AIMessage(
            content=(
                '{"segments":[{"text":"Three retries are allowed.",'
                '"citation_ids":["chunk-1"]}]}'
            )
        )
    )
    adapter._model = fake  # type: ignore[assignment]
    messages = [
        ModelMessage(role="user", content="x"),
        ModelMessage(
            role="tool",
            name="search_documents",
            tool_call_id="call-1",
            content=(
                '{"ok":true,"payload":{"sources":['
                '{"chunk_id":"chunk-1"},{"chunk_id":"chunk-2"}]}}'
            ),
        ),
    ]

    result = adapter.invoke_structured(messages, StructuredAnswer)

    assert result.segments[0].citation_ids == ["chunk-1"]
    assert fake.response_format is not None
    citation_schema = fake.response_format["schema"]["$defs"]["GeneratedSegment"]["properties"][
        "citation_ids"
    ]
    assert citation_schema["items"] == {
        "type": "string",
        "enum": ["chunk-1", "chunk-2"],
    }
    assert "maxItems" not in citation_schema


def test_structured_response_fails_closed_on_empty_model_content() -> None:
    adapter = _adapter()
    adapter._model = _FakeModel(AIMessage(content=""))  # type: ignore[assignment]

    with pytest.raises(ValueError, match="empty structured response"):
        adapter.invoke_structured(
            [ModelMessage(role="user", content="x")],
            StructuredAnswer,
        )


def test_structured_response_fails_closed_on_invalid_json() -> None:
    adapter = _adapter()
    adapter._model = _FakeModel(AIMessage(content="not-json"))  # type: ignore[assignment]

    with pytest.raises(ValueError):
        adapter.invoke_structured(
            [ModelMessage(role="user", content="x")],
            StructuredAnswer,
        )
