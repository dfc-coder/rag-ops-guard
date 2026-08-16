from __future__ import annotations

from typing import Any

from rag_ops_guard.agent.tools import SearchDocumentsTool
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports.interfaces import Tool, ToolResult
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult


class WeakKnowledge:
    def search(self, query: str, _context: QueryContext, **_kwargs: Any) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(
            dense=[], lexical=[], fused=[], admitted=[], relevance=0.42, supported=False
        )


def test_tool_port_has_required_surface() -> None:
    """SPEC-1a.1"""
    tool = SearchDocumentsTool(WeakKnowledge())  # type: ignore[arg-type]
    assert isinstance(tool.name, str) and tool.name
    assert isinstance(tool.description, str) and tool.description
    assert tool.schema()["type"] == "object"
    assert isinstance(tool, Tool)


def test_search_tool_returns_typed_result() -> None:
    """SPEC-1a.2"""
    tool = SearchDocumentsTool(WeakKnowledge())  # type: ignore[arg-type]
    result = tool.invoke({"query": "reintentos"})
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.payload["query"] == "reintentos"


def test_search_tool_does_not_decide_grounding() -> None:
    """SPEC-1a.5"""
    tool = SearchDocumentsTool(WeakKnowledge())  # type: ignore[arg-type]
    result = tool.invoke({"query": "pizza"})
    assert result.ok is True
    assert "supported" not in result.payload
    assert "grounded" not in result.payload
