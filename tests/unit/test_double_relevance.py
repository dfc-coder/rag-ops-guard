from __future__ import annotations

from typing import Any

from rag_ops_guard.agent.conversation import ConversationAgent
from rag_ops_guard.domain.models import GeneratedSegment, QueryContext, StructuredAnswer
from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, KnowledgeSearchResult
from tests.fixtures.builders import evidence
from tests.fixtures.fakes import FakeEmbeddingProvider, FakeObjectStore, FakeReranker, FakeVectorStore


class DropResolver:
    def resolve(self, _items: list[Any], _context: QueryContext, limit: int = 5) -> list[Any]:
        del limit
        return []


class KeepResolver:
    def resolve(self, items: list[Any], _context: QueryContext, limit: int = 5) -> list[Any]:
        return items[:limit]


def _search(resolver: Any) -> KnowledgeSearch:
    item = evidence(text="Widget policy contains the internal deployment rules.")
    item.chunk.title = "Widget Policy"
    objects = FakeObjectStore()
    objects.put_text(
        f"chunks/{item.chunk.logical_id}/{item.chunk.version}/chunk-000.json",
        item.chunk.model_dump_json(),
    )
    return KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[item]),
        objects=objects,
        resolver=resolver,
        reranker=FakeReranker(default_score=0.91, default_relevant=True),
        candidate_k=5,
        context_k=2,
        min_relevance=0.5,
    )


def test_domain_relevance_is_measured_before_governance_resolution() -> None:
    """SPEC-4.1 / SPEC-4.2"""
    result = _search(DropResolver()).search("widget policy", QueryContext())

    assert result.domain_relevance == 0.91
    assert result.grounded_relevance == 0.0
    assert result.admitted == []


def test_grounded_relevance_is_measured_after_resolution() -> None:
    """SPEC-4.2"""
    result = _search(KeepResolver()).search("widget policy", QueryContext())

    assert result.domain_relevance == 0.91
    assert result.grounded_relevance == 0.91
    assert result.supported is True


class ProbeKnowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, _context: QueryContext, **kwargs: Any) -> KnowledgeSearchResult:
        self.calls.append((query, str(kwargs.get("query_mode"))))
        return KnowledgeSearchResult(
            dense=[],
            lexical=[],
            fused=[],
            admitted=[],
            relevance=0.12,
            supported=False,
        )


class EmptyCatalog:
    def render(self, _question: str, _context: QueryContext) -> str:
        return "No documents"


class DirectModel:
    def bind_tools(self, _tools: list[Any]) -> DirectModel:
        return self

    def invoke(self, _messages: list[ModelMessage]) -> ModelTurn:
        return ModelTurn(content="Eight planets orbit the Sun.")

    def invoke_structured(self, _messages: list[ModelMessage], _schema: type[Any]) -> StructuredAnswer:
        return StructuredAnswer(
            segments=[GeneratedSegment(text="Eight planets orbit the Sun.", citation_ids=[])]
        )


def test_every_turn_records_a_non_tool_probe_without_forcing_document_tool_use() -> None:
    """SPEC-4.3 / SPEC-4.4"""
    knowledge = ProbeKnowledge()
    agent = ConversationAgent(
        knowledge=knowledge,  # type: ignore[arg-type]
        catalog=EmptyCatalog(),  # type: ignore[arg-type]
        model=DirectModel(),
    )

    result = agent.invoke(
        "How many planets are in the solar system?",
        thread_id="probe",
        context=QueryContext(),
    )

    assert knowledge.calls == [
        ("How many planets are in the solar system?", "probe"),
    ]
    assert result.tool_calls == 0
    assert result.domain_relevance_score == 0.12
    assert result.grounded_relevance_score == 0.0
