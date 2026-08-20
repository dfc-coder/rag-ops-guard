from __future__ import annotations

from typing import Any, TypeVar

from rag_ops_guard.agent.conversation import ConversationAgent, ConversationResponse
from rag_ops_guard.agent.tools import SearchDocumentsTool
from rag_ops_guard.domain.models import QueryContext, QueryStatus
from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool

T = TypeVar("T")


class DirectModel:
    def bind_tools(self, _tools: list[Tool]) -> DirectModel:
        return self

    def invoke(self, _messages: list[ModelMessage]) -> ModelTurn:
        return ModelTurn(content="I cannot determine that from the available information.")

    def invoke_structured(self, _messages: list[ModelMessage], schema: type[T]) -> T:
        return schema.model_validate(  # type: ignore[attr-defined,no-any-return]
            {
                "segments": [
                    {
                        "text": "I cannot determine that from the available information.",
                        "citation_ids": [],
                    }
                ]
            }
        )


class MustNotSearchKnowledge:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("QueryContext must not force document retrieval")


def test_scoped_context_does_not_force_document_search_when_model_skips_tools() -> None:
    knowledge = MustNotSearchKnowledge()
    search_tool = SearchDocumentsTool(knowledge)  # type: ignore[arg-type]
    agent = ConversationAgent(model=DirectModel(), tools=[search_tool])

    result = agent.invoke(
        "A Calypso submission timed out. What should I verify before manual replay?",
        thread_id="scoped-no-fallback",
        context=QueryContext(system="payments", environment="production"),
    )

    assert isinstance(result, ConversationResponse)
    assert result.status == QueryStatus.ANSWERED_UNGROUNDED
    assert result.route == "chat"
    assert result.tool_calls == 0
    assert result.citations == ()
    assert knowledge.calls == 0
