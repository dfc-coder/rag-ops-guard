from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import rag_ops_guard.agent.react_agent as react_agent_module
from rag_ops_guard.agent.react_agent import _model_prompt_messages, search_knowledge
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult


class BrokenSearch:
    def search(self, *args: Any, **kwargs: Any) -> KnowledgeSearchResult:
        del args, kwargs
        raise RuntimeError("backend unavailable")


class CapturingSearch:
    def __init__(self) -> None:
        self.query: str | None = None
        self.ranking_query: str | None = None

    def search(
        self,
        query: str,
        _context: Any,
        *,
        query_mode: str,
        ranking_query: str | None = None,
    ) -> KnowledgeSearchResult:
        assert query_mode == "knowledge"
        self.query = query
        self.ranking_query = ranking_query
        return KnowledgeSearchResult(dense=[], lexical=[], fused=[], admitted=[], supported=False)


def test_search_tool_converts_backend_exception_to_unsupported_payload(monkeypatch) -> None:
    monkeypatch.setattr(react_agent_module, "knowledge_search", lambda: BrokenSearch())

    payload = json.loads(search_knowledge.invoke({"query": "Calypso retry policy"}))

    assert payload["supported"] is False
    assert payload["reason"] == "retrieval_error"
    assert "backend unavailable" not in payload["message"]


def test_search_tool_passes_literal_ranking_query_to_retrieval(monkeypatch) -> None:
    backend = CapturingSearch()
    monkeypatch.setattr(react_agent_module, "knowledge_search", lambda: backend)

    payload = json.loads(
        search_knowledge.invoke(
            {
                "query": "Calypso retries. Y despues del tercero?",
                "ranking_query": "Y despues del tercero?",
            }
        )
    )

    assert backend.query == "Calypso retries. Y despues del tercero?"
    assert backend.ranking_query == "Y despues del tercero?"
    assert payload["supported"] is False
    assert payload["reason"] == "no_admitted_evidence"


def test_model_prompt_removes_tool_protocol_and_evidence_from_older_turns() -> None:
    old_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {"query": "old production fact"},
                "id": "old-search",
                "type": "tool_call",
            }
        ],
    )
    old_tool_result = ToolMessage(
        content='{"supported": true, "sources": [{"text": "STALE PRODUCTION EVIDENCE"}]}',
        name="search_knowledge",
        tool_call_id="old-search",
    )
    messages = [
        HumanMessage(content="old question"),
        old_tool_call,
        old_tool_result,
        AIMessage(content="old grounded answer"),
        HumanMessage(content="new unrelated question"),
    ]

    filtered = _model_prompt_messages(messages)

    assert old_tool_call not in filtered
    assert old_tool_result not in filtered
    assert any(
        isinstance(message, AIMessage) and message.content == "old grounded answer"
        for message in filtered
    )
    assert filtered[-1] == messages[-1]


def test_model_prompt_keeps_current_turn_tool_protocol() -> None:
    current_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {"query": "current fact"},
                "id": "current-search",
                "type": "tool_call",
            }
        ],
    )
    current_result = ToolMessage(
        content='{"supported": true, "sources": [{"text": "CURRENT EVIDENCE"}]}',
        name="search_knowledge",
        tool_call_id="current-search",
    )
    messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="current internal question"),
        current_call,
        current_result,
    ]

    filtered = _model_prompt_messages(messages)

    assert current_call in filtered
    assert current_result in filtered
