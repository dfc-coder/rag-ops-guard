from __future__ import annotations

import json
from typing import Any, TypeVar

from rag_ops_guard.agent.conversation import ConversationAgent, ConversationResponse
from rag_ops_guard.domain.models import QueryContext, QueryStatus
from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult
from tests.fixtures.builders import evidence, metadata

T = TypeVar("T")
CHUNK_ID = "calypso-timeout-runbook:1.0:000:deadbeef"


class RefusingModel:
    def bind_tools(self, _tools: list[Tool]) -> RefusingModel:
        return self

    def invoke(self, _messages: list[ModelMessage]) -> ModelTurn:
        return ModelTurn(content="I cannot access the internal runbook.")

    def invoke_structured(self, messages: list[ModelMessage], schema: type[T]) -> T:
        tool_messages = [
            message
            for message in messages
            if message.role == "tool" and message.name == "search_documents"
        ]
        assert tool_messages, "scoped internal query must reach search_documents"
        payload = json.loads(tool_messages[-1].content)["payload"]
        assert payload["sources"][0]["chunk_id"] == CHUNK_ID
        return schema.model_validate(  # type: ignore[attr-defined,no-any-return]
            {
                "segments": [
                    {
                        "text": "Verify the transaction state before manual replay.",
                        "citation_ids": [CHUNK_ID],
                    }
                ]
            }
        )


class FakeKnowledge:
    def search(self, _query: str, _context: QueryContext, **_kwargs: Any) -> KnowledgeSearchResult:
        meta = metadata(
            doc_id="calypso-timeout-runbook-v1",
            logical_id="calypso-timeout-runbook",
            version="1.0",
            authority=100,
        )
        meta.title = "Calypso Timeout Runbook"
        item = evidence(
            meta=meta,
            text="Verify the transaction state before manual replay.",
            distance=0.01,
        )
        item.chunk.id = CHUNK_ID
        item.chunk.title = "Calypso Timeout Runbook"
        return KnowledgeSearchResult(
            dense=[item],
            lexical=[item],
            fused=[item],
            admitted=[item],
            relevance=0.99,
            supported=True,
        )


class FakeCatalog:
    def render(self, _question: str, _context: QueryContext) -> str:
        return "Calypso Timeout Runbook"


def test_scoped_context_falls_back_to_document_search_when_model_skips_tools() -> None:
    agent = ConversationAgent(
        knowledge=FakeKnowledge(),  # type: ignore[arg-type]
        catalog=FakeCatalog(),  # type: ignore[arg-type]
        model=RefusingModel(),
    )

    result = agent.invoke(
        "A Calypso submission timed out. What should I verify before manual replay?",
        thread_id="scoped-fallback",
        context=QueryContext(system="payments", environment="production"),
    )

    assert isinstance(result, ConversationResponse)
    assert result.status == QueryStatus.ANSWERED_GROUNDED
    assert result.route == "knowledge"
    assert result.tool_calls == 1
    assert len(result.citations) == 1
    assert result.citations[0].logical_id == "calypso-timeout-runbook"
