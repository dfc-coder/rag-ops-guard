from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag_ops_guard.agent.semantic_router import (
    ContextRelation,
    SemanticConversationContext,
    SemanticTurnResolver,
    TurnDecision,
    TurnOperation,
)


class FakeBoundModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> AIMessage:
        self.calls.append(messages)
        return self.response


def _context() -> SemanticConversationContext:
    return SemanticConversationContext(
        has_grounded_topic=True,
        grounded_topic="payments retry policy",
        last_grounded_query="How many retries are allowed?",
        has_active_evidence=True,
        last_user_message="How many retries are allowed?",
        last_assistant_message="Three retries are allowed.",
        system_filter=None,
        environment_filter="production",
        api_version_filter=None,
    )


def _response(args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": TurnDecision.__name__,
                "args": args,
                "id": "turn-decision-1",
                "type": "tool_call",
            }
        ],
    )


def test_resolver_parses_valid_structured_decision() -> None:
    model = FakeBoundModel(
        _response(
            {
                "requires_grounding": True,
                "relation_to_context": "same",
                "operation": "answer",
                "standalone_query": "What happens after the third retry fails?",
            }
        )
    )
    resolver = SemanticTurnResolver(model)

    decision = resolver.resolve("What happens next?", _context())

    assert decision.requires_grounding is True
    assert decision.relation_to_context == ContextRelation.SAME
    assert decision.operation == TurnOperation.ANSWER
    assert decision.standalone_query == "What happens after the third retry fails?"
    assert len(model.calls) == 1


def test_resolver_rejects_grounded_decision_without_standalone_query() -> None:
    model = FakeBoundModel(
        _response(
            {
                "requires_grounding": True,
                "relation_to_context": "same",
                "operation": "answer",
                "standalone_query": None,
            }
        )
    )
    resolver = SemanticTurnResolver(model)

    with pytest.raises(ValueError, match="omitted standalone_query"):
        resolver.resolve("What happens next?", _context())


def test_resolver_normalizes_query_away_for_non_grounded_decision() -> None:
    model = FakeBoundModel(
        _response(
            {
                "requires_grounding": False,
                "relation_to_context": "none",
                "operation": "answer",
                "standalone_query": "should be ignored",
            }
        )
    )
    resolver = SemanticTurnResolver(model)

    decision = resolver.resolve("Write Fibonacci in Python", _context())

    assert decision.requires_grounding is False
    assert decision.standalone_query is None


def test_resolver_rejects_multiple_tool_calls() -> None:
    response = _response(
        {
            "requires_grounding": False,
            "relation_to_context": "none",
            "operation": "answer",
            "standalone_query": None,
        }
    )
    response.tool_calls.append(response.tool_calls[0].copy())
    resolver = SemanticTurnResolver(FakeBoundModel(response))

    with pytest.raises(ValueError, match="expected exactly 1"):
        resolver.resolve("hello", _context())
