from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import APITimeoutError, LengthFinishReasonError
from pydantic import ValidationError

from rag_ops_guard.agent.catalog import KnowledgeCatalog
from rag_ops_guard.agent.responses import (
    capabilities_response,
    insufficient_evidence_response,
    out_of_scope_response,
)
from rag_ops_guard.agent.router import Route, RouteDecision, SemanticRouter
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


class ConversationFocus(TypedDict):
    query: str
    source_titles: list[str]


class AgentState(MessagesState, total=False):
    request_id: str
    context: QueryContext
    route: Route
    route_confidence: float
    route_margin: float
    route_scores: dict[str, float]
    evidence: list[Evidence]
    answer: str | None
    citations: list[Citation]
    status: QueryStatus
    timings_ms: dict[str, float]
    retrieval_query: str
    rewritten_query: str | None
    relevance_score: float
    focus: ConversationFocus


class ConversationalAgent:
    """Bounded conversational agent with trusted memory and grounded retrieval."""

    def __init__(
        self,
        *,
        chat: ChatModel,
        router: SemanticRouter,
        knowledge: KnowledgeSearch,
        catalog: KnowledgeCatalog,
        relevance_threshold: float = 0.4,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        self._chat = chat
        self._router = router
        self._knowledge = knowledge
        self._catalog = catalog
        self._relevance_threshold = relevance_threshold
        self._graph = self._build_graph(checkpointer or InMemorySaver())

    def _build_graph(self, checkpointer: InMemorySaver) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("chat", self._generate_chat)
        builder.add_node("capabilities", self._capabilities)
        builder.add_node("catalog", self._catalog_response)
        builder.add_node("out_of_scope", self._out_of_scope)
        builder.add_node("uncertain", self._uncertain)
        builder.add_node("search_knowledge", self._search_knowledge)
        builder.add_node("grounded_answer", self._generate_grounded_answer)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {
                "chat": "chat",
                "capabilities": "capabilities",
                "catalog": "catalog",
                "knowledge": "search_knowledge",
                "out_of_scope": "out_of_scope",
                "uncertain": "uncertain",
            },
        )
        builder.add_edge("search_knowledge", "grounded_answer")
        for node in ("chat", "capabilities", "catalog", "out_of_scope", "uncertain"):
            builder.add_edge(node, END)
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
            "evidence": [],
            "answer": None,
            "timings_ms": {},
            "retrieval_query": "",
            "rewritten_query": None,
            "relevance_score": 0.0,
            "route_scores": {},
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
            route_confidence=result.get("route_confidence"),
            route_margin=result.get("route_margin"),
            relevance_score=result.get("relevance_score"),
            retrieval_query=result.get("retrieval_query") or None,
            rewritten_query=result.get("rewritten_query"),
        )

    def refresh_knowledge(self) -> None:
        self._knowledge.refresh()

    def _route(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        question = _latest_user_message(state["messages"])
        decision: RouteDecision = self._router.route(question)
        route = decision.route
        if route == "uncertain" and state.get("focus"):
            route = "knowledge"
        route_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "agent_routed",
            extra={
                "request_id": state["request_id"],
                "question": question,
                "route": route,
                "route_score": decision.score,
                "route_margin": decision.margin,
                "route_scores": decision.scores,
                "route_ms": route_ms,
            },
        )
        return {
            "route": route,
            "route_confidence": decision.score,
            "route_margin": decision.margin,
            "route_scores": decision.scores,
            "timings_ms": _timings(state, route=route_ms),
        }

    def _search_knowledge(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        question = _latest_user_message(state["messages"])
        result = self._safe_search(question, state)
        retrieval_query = question
        rewritten_query: str | None = None
        rewrite_ms = 0.0

        focus = state.get("focus")
        if result.relevance < self._relevance_threshold and focus:
            rewrite_started = perf_counter()
            try:
                candidate = self._chat.rewrite_query(
                    current_question=question,
                    previous_query=focus["query"],
                    source_titles=focus["source_titles"],
                )
                rewritten_query = candidate.strip()
            except (APITimeoutError, LengthFinishReasonError, ValidationError) as exc:
                QUERY_LOGGER.warning(
                    "query_rewrite_failed",
                    extra={"request_id": state["request_id"], "reason": type(exc).__name__},
                )
                rewritten_query = None
            rewrite_ms = round((perf_counter() - rewrite_started) * 1000, 2)

            if rewritten_query and rewritten_query != question:
                rewritten_result = self._safe_search(rewritten_query, state)
                if rewritten_result.relevance > result.relevance:
                    result = rewritten_result
                    retrieval_query = rewritten_query

        search_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "knowledge_search_completed",
            extra={
                "request_id": state["request_id"],
                "original_question": question,
                "retrieval_query": retrieval_query,
                "rewritten_query": rewritten_query or "",
                "relevance": result.relevance,
                "dense_titles": [item.chunk.title for item in result.dense[:5]],
                "lexical_titles": [item.chunk.title for item in result.lexical[:5]],
                "admitted_titles": [item.chunk.title for item in result.admitted],
                "search_ms": search_ms,
                "rewrite_ms": rewrite_ms,
            },
        )
        updates: dict[str, float] = {"search": search_ms}
        if rewrite_ms:
            updates["rewrite"] = rewrite_ms
        return {
            "evidence": result.admitted,
            "retrieval_query": retrieval_query,
            "rewritten_query": rewritten_query,
            "relevance_score": result.relevance,
            "timings_ms": _timings(state, **updates),
        }

    def _safe_search(self, query: str, state: AgentState) -> KnowledgeSearchResult:
        try:
            return self._knowledge.search(query, state["context"])
        except EvidenceConflictError as exc:
            QUERY_LOGGER.warning(
                "evidence_conflict",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            return KnowledgeSearchResult(dense=[], lexical=[], fused=[], admitted=[], relevance=0.0)

    def _generate_chat(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        answer = self._chat.generate_chat(_recent_messages(state["messages"]))
        generation_ms = round((perf_counter() - started) * 1000, 2)
        return self._answered_update(state, answer, generation_ms=generation_ms)

    def _capabilities(self, state: AgentState) -> dict[str, Any]:
        answer = capabilities_response(_latest_user_message(state["messages"]))
        return self._answered_update(state, answer)

    def _catalog_response(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        question = _latest_user_message(state["messages"])
        answer = self._catalog.render(question, state["context"])
        catalog_ms = round((perf_counter() - started) * 1000, 2)
        return self._answered_update(state, answer, catalog_ms=catalog_ms)

    def _out_of_scope(self, state: AgentState) -> dict[str, Any]:
        answer = out_of_scope_response(_latest_user_message(state["messages"]))
        return self._answered_update(state, answer)

    def _uncertain(self, state: AgentState) -> dict[str, Any]:
        answer = out_of_scope_response(_latest_user_message(state["messages"]))
        return self._answered_update(state, answer)

    def _answered_update(
        self,
        state: AgentState,
        answer: str,
        *,
        generation_ms: float | None = None,
        catalog_ms: float | None = None,
    ) -> dict[str, Any]:
        timing_updates: dict[str, float] = {}
        if generation_ms is not None:
            timing_updates["generation"] = generation_ms
        if catalog_ms is not None:
            timing_updates["catalog"] = catalog_ms
        return {
            "messages": [AIMessage(content=answer)],
            "status": QueryStatus.ANSWERED,
            "answer": answer,
            "citations": [],
            "timings_ms": _timings(state, **timing_updates),
        }

    def _generate_grounded_answer(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        question = _latest_user_message(state["messages"])
        relevance = state.get("relevance_score", 0.0)
        if not evidence or relevance < self._relevance_threshold:
            answer = insufficient_evidence_response(question)
            return {
                "messages": [AIMessage(content=answer)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": answer,
                "citations": [],
            }

        started = perf_counter()
        try:
            grounded = self._chat.generate_answer(answer_prompt(question, evidence), history=None)
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
            answer = insufficient_evidence_response(question)
            return {
                "messages": [AIMessage(content=answer)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": answer,
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
            answer = insufficient_evidence_response(question)
            return {
                "messages": [AIMessage(content=answer)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": answer,
                "citations": [],
                "timings_ms": _timings(state, generation=generation_ms),
            }

        trusted_query = state.get("retrieval_query") or question
        focus: ConversationFocus = {
            "query": trusted_query,
            "source_titles": [citation.title for citation in citations],
        }
        return {
            "messages": [AIMessage(content=grounded.answer)],
            "status": QueryStatus.ANSWERED,
            "answer": grounded.answer,
            "citations": citations,
            "focus": focus,
            "timings_ms": _timings(state, generation=generation_ms),
        }


def _timings(state: AgentState, **updates: float) -> dict[str, float]:
    return {**state.get("timings_ms", {}), **updates}


def _recent_messages(messages: Sequence[BaseMessage], limit: int = 8) -> list[BaseMessage]:
    return list(messages[-limit:])


def _latest_user_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("conversation has no user message")


def _expand_citation_refs(citation_ids: list[str], evidence: list[Evidence]) -> list[str]:
    lookup = {f"E{index}": item.chunk.id for index, item in enumerate(evidence, start=1)}
    lookup.update({item.chunk.id: item.chunk.id for item in evidence})
    return [lookup.get(citation_id, citation_id) for citation_id in citation_ids]
