from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_ops_guard.adapters.llm.openai_tool_calling import (
    _content_text,
    _json_content,
    _structured_prompt,
    _to_langchain_messages,
)
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


def test_json_content_accepts_raw_json_and_strips_code_fences() -> None:
    assert _json_content('{"segments": []}') == '{"segments": []}'
    assert _json_content('```json\n{"segments": []}\n```') == '{"segments": []}'


def test_structured_prompt_carries_exact_pydantic_schema_instruction() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    messages = [ModelMessage(role="system", content="system"), ModelMessage(role="user", content="x")]

    prompted = _structured_prompt(messages, schema)

    assert prompted[0].role == "system"
    assert "Return only one JSON object" in prompted[0].content
    assert json.dumps(schema, ensure_ascii=False, separators=(",", ":")) in prompted[0].content
