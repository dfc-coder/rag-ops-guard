from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import APITimeoutError, LengthFinishReasonError
from pydantic import ValidationError

from rag_ops_guard.agent.router import Route, SemanticRouter
from rag_ops_guard.domain.errors import CitationValidationError, EvidenceConflictError
from rag_ops_guard.domain.models import (
    Citation,
    Evidence,
    QueryContext,
    QueryRequest,
    QueryResponse,
    QueryStatus,
)
from rag_ops_guard.graph.prompts import answer_prompt
from rag_ops_guard.observability.runtime import QUERY_LOGGER
from rag_ops_guard.ports import ChatModel
from rag_ops_guard.retrieval.citations import validate_citations
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, KnowledgeSearchResult


class AgentState(MessagesState, total=False):
    request_id: str
    context: QueryContext
    route: Route
    evidence: list[Evidence]
    answer: str | None
    citations: list[Citation]
    status: QueryStatus
    timings_ms: dict[str, float]


class ConversationalAgent:
    """Bounded conversational agent with optional grounded knowledge retrieval."""

    def __init__(
        self,
        *,
        chat: ChatModel,
        router: SemanticRouter,
        knowledge: KnowledgeSearch,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        self._chat = chat
        self._router = router
        self._knowledge = knowledge
        self._graph = self._build_graph(checkpointer or InMemorySaver())

    def _build_graph(self, checkpointer: InMemorySaver) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("search_knowledge", self._search_knowledge)
        builder.add_node("chat", self._generate_chat)
        builder.add_node("grounded_answer", self._generate_grounded_answer)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {"chat": "chat", "knowledge": "search_knowledge"},
        )
        builder.add_edge("search_knowledge", "grounded_answer")
        builder.add_edge("chat", END)
        builder.add_edge("grounded_answer", END)
        return builder.compile(checkpointer=checkpointer)

    def invoke(self, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        request_id = str(uuid4())
        thread_id = request.thread_id or request_id
        state: AgentState = {
            "messages": [HumanMessage(content=request.question.strip())],
            "request_id": request_id,
            "context": request.context,
            "citations": [],
            "timings_ms": {},
        }
        result = self._graph.invoke(
            state,
            config={
                "configurable": {"thread_id": thread_id},
                "run_name": "agent-turn",
                "tags": ["rag-ops-guard", "conversational-agent"],
                "metadata": {
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "system": request.context.system or "",
                    "environment": request.context.environment or "",
                    "api_version": request.context.api_version or "",
                },
            },
        )
        timings = dict(result.get("timings_ms", {}))
        timings["total"] = round((perf_counter() - started) * 1000, 2)
        return QueryResponse(
            request_id=request_id,
            status=result["status"],
            route=result["route"],
            answer=result.get("answer"),
            citations=result.get("citations", []),
            timings_ms=timings,
        )

    def refresh_knowledge(self) -> None:
        self._knowledge.refresh()

    def _route(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        route_text = _recent_user_context(state["messages"])
        route = self._router.route(route_text)
        route_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "agent_routed",
            extra={
                "request_id": state["request_id"],
                "route": route,
                "route_ms": route_ms,
            },
        )
        return {
            "route": route,
            "timings_ms": _timings(state, route=route_ms),
        }

    def _search_knowledge(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        query = _recent_user_context(state["messages"])
        try:
            result = self._knowledge.search(query, state["context"])
        except EvidenceConflictError as exc:
            QUERY_LOGGER.warning(
                "evidence_conflict",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            result = KnowledgeSearchResult(dense=[], lexical=[], fused=[], admitted=[])
        search_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "knowledge_search_completed",
            extra={
                "request_id": state["request_id"],
                "dense_titles": [item.chunk.title for item in result.dense[:5]],
                "lexical_titles": [item.chunk.title for item in result.lexical[:5]],
                "admitted_titles": [item.chunk.title for item in result.admitted],
                "search_ms": search_ms,
            },
        )
        return {
            "evidence": result.admitted,
            "timings_ms": _timings(state, search=search_ms),
        }

    def _generate_chat(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        answer = self._chat.generate_chat(_recent_messages(state["messages"]))
        generation_ms = round((perf_counter() - started) * 1000, 2)
        return {
            "messages": [AIMessage(content=answer)],
            "status": QueryStatus.ANSWERED,
            "answer": answer,
            "citations": [],
            "timings_ms": _timings(state, generation=generation_ms),
        }

    def _generate_grounded_answer(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        evidence = state.get("evidence", [])
        question = _latest_user_message(state["messages"])
        history = _prior_messages(state["messages"])
        try:
            grounded = self._chat.generate_answer(
                answer_prompt(question, evidence),
                history=history,
            )
        except (APITimeoutError, LengthFinishReasonError, ValidationError) as exc:
            generation_ms = round((perf_counter() - started) * 1000, 2)
            QUERY_LOGGER.warning(
                "generation_failed",
                extra={
                    "request_id": state["request_id"],
                    "reason": type(exc).__name__,
                    "generation_ms": generation_ms,
                },
            )
            return {
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": None,
                "citations": [],
                "timings_ms": _timings(state, generation=generation_ms),
            }

        generation_ms = round((perf_counter() - started) * 1000, 2)
        if grounded.status != "answered":
            return {
                "messages": [AIMessage(content=grounded.answer)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": grounded.answer,
                "citations": [],
                "timings_ms": _timings(state, generation=generation_ms),
            }

        try:
            citation_ids = _expand_citation_refs(grounded.citation_ids, evidence)
            citations = validate_citations(citation_ids, evidence)
        except CitationValidationError as exc:
            QUERY_LOGGER.warning(
                "invalid_generated_citation",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            return {
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": None,
                "citations": [],
                "timings_ms": _timings(state, generation=generation_ms),
            }

        return {
            "messages": [AIMessage(content=grounded.answer)],
            "status": QueryStatus.ANSWERED,
            "answer": grounded.answer,
            "citations": citations,
            "timings_ms": _timings(state, generation=generation_ms),
        }


def _timings(state: AgentState, **updates: float) -> dict[str, float]:
    return {**state.get("timings_ms", {}), **updates}


def _recent_messages(messages: Sequence[BaseMessage], limit: int = 8) -> list[BaseMessage]:
    return list(messages[-limit:])


def _prior_messages(messages: Sequence[BaseMessage], limit: int = 6) -> list[BaseMessage]:
    return list(messages[:-1][-limit:])


def _latest_user_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("conversation has no user message")


def _recent_user_context(messages: Sequence[BaseMessage], limit: int = 3) -> str:
    user_messages = [
        str(message.content) for message in messages if isinstance(message, HumanMessage)
    ]
    return "\n".join(user_messages[-limit:])


def _expand_citation_refs(citation_ids: list[str], evidence: list[Evidence]) -> list[str]:
    lookup = {f"E{index}": item.chunk.id for index, item in enumerate(evidence, start=1)}
    lookup.update({item.chunk.id: item.chunk.id for item in evidence})
    return [lookup.get(citation_id, citation_id) for citation_id in citation_ids]
