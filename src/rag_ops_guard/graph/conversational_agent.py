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
    safety_blocked_response,
)
from rag_ops_guard.agent.router import Route, RouteDecision, SemanticRouter
from rag_ops_guard.agent.safety import SafetyGuard
from rag_ops_guard.domain.errors import EvidenceConflictError
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
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, KnowledgeSearchResult, QueryMode


class ConversationFocus(TypedDict):
    query: str
    source_titles: list[str]
    context: dict[str, str | None]


class AgentState(MessagesState, total=False):
    request_id: str
    context: QueryContext
    route: Route
    route_confidence: float
    route_margin: float
    route_scores: dict[str, float]
    evidence: list[Evidence]
    evidence_supported: bool
    answer: str | None
    citations: list[Citation]
    status: QueryStatus
    timings_ms: dict[str, float]
    retrieval_query: str
    rewritten_query: str | None
    relevance_score: float
    focus: ConversationFocus | None


class ConversationalAgent:
    """Bounded conversational agent with trusted memory and grounded retrieval."""

    def __init__(
        self,
        *,
        chat: ChatModel,
        router: SemanticRouter,
        knowledge: KnowledgeSearch,
        catalog: KnowledgeCatalog,
        checkpointer: InMemorySaver | None = None,
        safety: SafetyGuard | None = None,
    ) -> None:
        self._chat = chat
        self._router = router
        self._knowledge = knowledge
        self._catalog = catalog
        self._checkpointer = checkpointer or InMemorySaver()
        self._safety = safety or SafetyGuard()
        self._graph = self._build_graph(self._checkpointer)

    def _build_graph(self, checkpointer: InMemorySaver) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("chat", self._generate_chat)
        builder.add_node("capabilities", self._capabilities)
        builder.add_node("catalog", self._catalog_response)
        builder.add_node("out_of_scope", self._out_of_scope)
        builder.add_node("safety", self._safety_blocked)
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
                "uncertain": "search_knowledge",
                "safety": "safety",
            },
        )
        builder.add_conditional_edges(
            "search_knowledge",
            lambda state: state["route"],
            {
                "chat": "chat",
                "capabilities": "capabilities",
                "catalog": "catalog",
                "knowledge": "grounded_answer",
                "out_of_scope": "out_of_scope",
                "safety": "safety",
            },
        )
        for node in ("chat", "capabilities", "catalog", "out_of_scope", "safety"):
            builder.add_edge(node, END)
        builder.add_edge("grounded_answer", END)
        return builder.compile(checkpointer=checkpointer)

    def invoke(self, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        request_id = str(uuid4())
        thread_id = request.thread_id or request_id
        state: AgentState = {
            "messages": [HumanMessage(content=request.question)],
            "request_id": request_id,
            "context": request.context,
            "citations": [],
            "evidence": [],
            "evidence_supported": False,
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

    def clear_thread(self, thread_id: str) -> None:
        self._checkpointer.delete_thread(thread_id)

    def _route(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        question = _latest_user_message(state["messages"])
        if self._safety.blocked(question):
            route_ms = round((perf_counter() - started) * 1000, 2)
            return {
                "route": "safety",
                "route_confidence": 1.0,
                "route_margin": 1.0,
                "route_scores": {},
                "timings_ms": _timings(state, route=route_ms),
            }

        decision: RouteDecision = self._router.route(question)
        route_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "agent_routed",
            extra={
                "request_id": state["request_id"],
                "question": question,
                "route": decision.route,
                "route_score": decision.score,
                "route_margin": decision.margin,
                "route_scores": decision.scores,
                "route_ms": route_ms,
            },
        )
        return {
            "route": decision.route,
            "route_confidence": decision.score,
            "route_margin": decision.margin,
            "route_scores": decision.scores,
            "timings_ms": _timings(state, route=route_ms),
        }

    def _search_knowledge(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        original_route = state["route"]
        question = _latest_user_message(state["messages"])
        initial_mode: QueryMode = "probe" if original_route == "uncertain" else "knowledge"
        result = self._safe_search(question, state, query_mode=initial_mode)
        retrieval_query = question
        rewritten_query: str | None = None
        rewrite_ms = 0.0

        stored_focus = state.get("focus")
        focus = (
            stored_focus
            if stored_focus and _focus_compatible(stored_focus, state["context"])
            else None
        )
        if not result.supported and focus:
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
                rewritten_result = self._safe_search(
                    rewritten_query,
                    state,
                    query_mode="knowledge",
                )
                if rewritten_result.supported or rewritten_result.relevance > result.relevance:
                    result = rewritten_result
                    retrieval_query = rewritten_query

        resolved_route = original_route
        if original_route == "uncertain":
            resolved_route = (
                "knowledge" if result.supported else _best_control_route(state.get("route_scores", {}))
            )

        search_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "knowledge_search_completed",
            extra={
                "request_id": state["request_id"],
                "original_question": question,
                "original_route": original_route,
                "resolved_route": resolved_route,
                "retrieval_query": retrieval_query,
                "rewritten_query": rewritten_query or "",
                "query_mode": initial_mode,
                "relevance": result.relevance,
                "supported": result.supported,
                "reranker_scores": result.reranker_scores,
                "lexical_relevance": result.lexical_relevance,
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
            "route": resolved_route,
            "evidence": result.admitted,
            "evidence_supported": result.supported,
            "retrieval_query": retrieval_query,
            "rewritten_query": rewritten_query,
            "relevance_score": result.relevance,
            "timings_ms": _timings(state, **updates),
        }

    def _safe_search(
        self,
        query: str,
        state: AgentState,
        *,
        query_mode: QueryMode,
    ) -> KnowledgeSearchResult:
        try:
            return self._knowledge.search(query, state["context"], query_mode=query_mode)
        except EvidenceConflictError as exc:
            QUERY_LOGGER.warning(
                "evidence_conflict",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            return KnowledgeSearchResult(
                dense=[],
                lexical=[],
                fused=[],
                admitted=[],
                relevance=0.0,
                supported=False,
            )

    def _generate_chat(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        answer = self._chat.generate_chat(_recent_messages(state["messages"]))
        generation_ms = round((perf_counter() - started) * 1000, 2)
        return self._answered_update(state, answer, generation_ms=generation_ms)

    def _capabilities(self, state: AgentState) -> dict[str, Any]:
        answer = capabilities_response(_latest_user_message(state["messages"]))
        return self._answered_update(state, answer, clear_focus=True)

    def _catalog_response(self, state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        question = _latest_user_message(state["messages"])
        answer = self._catalog.render(question, state["context"])
        catalog_ms = round((perf_counter() - started) * 1000, 2)
        return self._answered_update(state, answer, catalog_ms=catalog_ms, clear_focus=True)

    def _out_of_scope(self, state: AgentState) -> dict[str, Any]:
        answer = out_of_scope_response(_latest_user_message(state["messages"]))
        return self._answered_update(state, answer, clear_focus=True)

    def _safety_blocked(self, state: AgentState) -> dict[str, Any]:
        answer = safety_blocked_response(_latest_user_message(state["messages"]))
        return {
            "messages": [AIMessage(content=answer)],
            "status": QueryStatus.SAFETY_BLOCKED,
            "answer": answer,
            "citations": [],
            "focus": None,
        }

    def _answered_update(
        self,
        state: AgentState,
        answer: str,
        *,
        generation_ms: float | None = None,
        catalog_ms: float | None = None,
        clear_focus: bool = False,
    ) -> dict[str, Any]:
        timing_updates: dict[str, float] = {}
        if generation_ms is not None:
            timing_updates["generation"] = generation_ms
        if catalog_ms is not None:
            timing_updates["catalog"] = catalog_ms
        result: dict[str, Any] = {
            "messages": [AIMessage(content=answer)],
            "status": QueryStatus.ANSWERED,
            "answer": answer,
            "citations": [],
            "timings_ms": _timings(state, **timing_updates),
        }
        if clear_focus:
            result["focus"] = None
        return result

    def _generate_grounded_answer(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence", [])
        question = _latest_user_message(state["messages"])
        if not evidence or not state.get("evidence_supported", False):
            answer = insufficient_evidence_response(question)
            return {
                "messages": [AIMessage(content=answer)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": answer,
                "citations": [],
                "focus": None,
            }

        started = perf_counter()
        try:
            answer = self._chat.generate_grounded_text(answer_prompt(question, evidence))
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
            fallback = insufficient_evidence_response(question)
            return {
                "messages": [AIMessage(content=fallback)],
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": fallback,
                "citations": [],
                "focus": None,
                "timings_ms": _timings(state, generation=generation_ms),
            }

        generation_ms = round((perf_counter() - started) * 1000, 2)
        citations = validate_citations(_admitted_source_ids(evidence), evidence)
        trusted_query = state.get("retrieval_query") or question
        focus: ConversationFocus = {
            "query": trusted_query,
            "source_titles": [citation.title for citation in citations],
            "context": _context_snapshot(state["context"]),
        }
        return {
            "messages": [AIMessage(content=answer)],
            "status": QueryStatus.ANSWERED,
            "answer": answer,
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


def _context_snapshot(context: QueryContext) -> dict[str, str | None]:
    return {
        "system": context.system,
        "environment": context.environment,
        "api_version": context.api_version,
    }


def _focus_compatible(focus: ConversationFocus, context: QueryContext) -> bool:
    return focus["context"] == _context_snapshot(context)


def _best_control_route(scores: dict[str, float]) -> Route:
    candidates: tuple[Route, ...] = ("capabilities", "catalog", "chat", "out_of_scope")
    ranked = [(route, scores.get(route, float("-inf"))) for route in candidates]
    best_route, best_score = max(ranked, key=lambda item: item[1])
    if best_score == float("-inf"):
        return "out_of_scope"
    return best_route


def _admitted_source_ids(evidence: list[Evidence]) -> list[str]:
    """Attach one valid citation per admitted document version, preserving rank order."""
    seen: set[tuple[str, str]] = set()
    citation_ids: list[str] = []
    for item in evidence:
        key = (item.chunk.logical_id, item.chunk.version)
        if key in seen:
            continue
        seen.add(key)
        citation_ids.append(item.chunk.id)
    return citation_ids
