from __future__ import annotations

import json
from typing import Any

import rag_ops_guard.agent.react_agent as react_agent_module
from rag_ops_guard.agent.react_agent import search_knowledge
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
