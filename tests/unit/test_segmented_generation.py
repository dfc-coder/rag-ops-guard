from __future__ import annotations

from typing import Any, TypeVar

from rag_ops_guard.agent.conversation import ConversationAgent
from rag_ops_guard.domain.models import QueryContext, QueryStatus, StructuredAnswer
from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool

T = TypeVar("T")


class StructuredModel:
    def __init__(self) -> None:
        self.structured_calls = 0

    def bind_tools(self, _tools: list[Tool]) -> StructuredModel:
        return self

    def invoke(self, _messages: list[ModelMessage]) -> ModelTurn:
        return ModelTurn(content="DRAFT TEXT MUST NOT BECOME THE PUBLIC RESPONSE")

    def invoke_structured(self, _messages: list[ModelMessage], schema: type[T]) -> T:
        self.structured_calls += 1
        return schema.model_validate(  # type: ignore[attr-defined,no-any-return]
            {"segments": [{"text": "Structured final answer", "citation_ids": []}]}
        )


def test_public_segments_come_from_structured_generation() -> None:
    """SPEC-1.6: finished prose is never split into segments after generation."""
    model = StructuredModel()
    agent = ConversationAgent(model=model, tools=[])

    response = agent.invoke(
        "Explain exponential backoff",
        thread_id="structured-direct",
        context=QueryContext(),
    )

    assert model.structured_calls == 1
    assert response.status is QueryStatus.ANSWERED_UNGROUNDED
    assert response.answer == "Structured final answer"
    assert [segment.text for segment in response.segments] == ["Structured final answer"]
    assert "DRAFT TEXT" not in response.answer


def test_structured_schema_is_the_canonical_final_response_shape() -> None:
    """SPEC-1.6"""
    schema = StructuredAnswer.model_json_schema()
    assert "segments" in schema["properties"]
