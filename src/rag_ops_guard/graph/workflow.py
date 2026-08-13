from __future__ import annotations

from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rag_ops_guard.domain.errors import CitationValidationError, EvidenceConflictError
from rag_ops_guard.domain.models import QueryRequest, QueryResponse, QueryStatus
from rag_ops_guard.graph.prompts import analysis_prompt, answer_prompt
from rag_ops_guard.graph.state import RagState
from rag_ops_guard.observability.runtime import QUERY_LOGGER
from rag_ops_guard.ports import ChatModel, EmbeddingProvider, VectorStore
from rag_ops_guard.retrieval.citations import validate_citations
from rag_ops_guard.retrieval.query_instruction import embedding_query
from rag_ops_guard.retrieval.resolver import EvidenceResolver


class RagWorkflow:
    def __init__(
        self,
        chat: ChatModel,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        resolver: EvidenceResolver,
        retrieval_top_k: int = 8,
        retrieval_context_k: int = 5,
    ) -> None:
        self._chat = chat
        self._embeddings = embeddings
        self._vectors = vectors
        self._resolver = resolver
        self._top_k = retrieval_top_k
        self._context_k = retrieval_context_k
        self._graph = self._build_graph()

    def invoke(self, request: QueryRequest) -> QueryResponse:
        state: RagState = {
            "request_id": str(uuid4()),
            "question": request.question.strip(),
            "context": request.context,
            "graph_path": [],
            "citations": [],
        }
        result = self._graph.invoke(state)
        return QueryResponse(
            request_id=result["request_id"],
            status=result["status"],
            answer=result.get("answer"),
            clarification_question=result.get("clarification_question"),
            citations=result.get("citations", []),
        )

    def _build_graph(self) -> Any:
        builder = StateGraph(RagState)
        builder.add_node("validate_request", self._validate_request)
        builder.add_node("analyze_query", self._analyze_query)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("resolve_evidence", self._resolve_evidence)
        builder.add_node("generate_grounded_answer", self._generate_grounded_answer)
        builder.add_node("validate_citations", self._validate_citations)
        builder.add_node("clarification_required", self._clarification_required)
        builder.add_node("safety_blocked", self._safety_blocked)
        builder.add_node("insufficient_evidence", self._insufficient_evidence)

        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "analyze_query")
        builder.add_conditional_edges("analyze_query", self._route_after_analysis)
        builder.add_edge("retrieve_evidence", "resolve_evidence")
        builder.add_conditional_edges("resolve_evidence", self._route_after_resolution)
        builder.add_conditional_edges("generate_grounded_answer", self._route_after_generation)
        builder.add_edge("validate_citations", END)
        builder.add_edge("clarification_required", END)
        builder.add_edge("safety_blocked", END)
        builder.add_edge("insufficient_evidence", END)
        return builder.compile()

    @staticmethod
    def _append_path(state: RagState, node: str) -> list[str]:
        return [*state.get("graph_path", []), node]

    def _validate_request(self, state: RagState) -> RagState:
        question = state["question"].strip()
        if not 3 <= len(question) <= 2000:
            raise ValueError("question must contain 3..2000 characters")
        return {"question": question, "graph_path": self._append_path(state, "validate_request")}

    def _analyze_query(self, state: RagState) -> RagState:
        analysis = self._chat.analyze_query(analysis_prompt(state["question"], state["context"]))
        update: RagState = {
            "normalized_question": analysis.normalized_question,
            "clarification_question": analysis.clarification_question,
            "safety_category": analysis.safety_category,
            "safety_blocked_message": analysis.safety_blocked_message,
            "insufficient_evidence_message": analysis.insufficient_evidence_message,
            "graph_path": self._append_path(state, "analyze_query"),
        }
        if analysis.safety_category != "normal":
            update["status"] = QueryStatus.SAFETY_BLOCKED
        elif analysis.requires_clarification:
            update["status"] = QueryStatus.CLARIFICATION_REQUIRED
        return update

    @staticmethod
    def _route_after_analysis(
        state: RagState,
    ) -> Literal["safety_blocked", "clarification_required", "retrieve_evidence"]:
        if state.get("status") == QueryStatus.SAFETY_BLOCKED:
            return "safety_blocked"
        if state.get("status") == QueryStatus.CLARIFICATION_REQUIRED:
            return "clarification_required"
        return "retrieve_evidence"

    def _retrieve_evidence(self, state: RagState) -> RagState:
        started = perf_counter()
        vector = self._embeddings.embed_query(embedding_query(state["normalized_question"]))
        evidence = self._vectors.query(vector, self._top_k)
        QUERY_LOGGER.info(
            "retrieval_completed",
            extra={
                "request_id": state["request_id"],
                "retrieved_documents": len(evidence),
                "retrieval_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "retrieved_evidence": evidence,
            "graph_path": self._append_path(state, "retrieve_evidence"),
        }

    def _resolve_evidence(self, state: RagState) -> RagState:
        try:
            resolved = self._resolver.resolve(
                state.get("retrieved_evidence", []),
                state["context"],
                limit=self._context_k,
            )
        except EvidenceConflictError as exc:
            QUERY_LOGGER.warning(
                "evidence_conflict",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            resolved = []
        update: RagState = {
            "resolved_evidence": resolved,
            "graph_path": self._append_path(state, "resolve_evidence"),
        }
        if not resolved:
            update["status"] = QueryStatus.INSUFFICIENT_EVIDENCE
        return update

    @staticmethod
    def _route_after_resolution(
        state: RagState,
    ) -> Literal["insufficient_evidence", "generate_grounded_answer"]:
        if state.get("status") == QueryStatus.INSUFFICIENT_EVIDENCE:
            return "insufficient_evidence"
        return "generate_grounded_answer"

    def _generate_grounded_answer(self, state: RagState) -> RagState:
        started = perf_counter()
        grounded = self._chat.generate_answer(
            answer_prompt(state["question"], state["resolved_evidence"])
        )
        QUERY_LOGGER.info(
            "generation_completed",
            extra={
                "request_id": state["request_id"],
                "generation_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        update: RagState = {
            "answer": grounded.answer,
            "graph_path": self._append_path(state, "generate_grounded_answer"),
        }
        if grounded.status == "insufficient_evidence":
            update["status"] = QueryStatus.INSUFFICIENT_EVIDENCE
            update["citations"] = []
            return update

        try:
            citations = validate_citations(grounded.citation_ids, state["resolved_evidence"])
        except CitationValidationError as exc:
            QUERY_LOGGER.warning(
                "invalid_generated_citation",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            update["status"] = QueryStatus.INSUFFICIENT_EVIDENCE
            update["citations"] = []
            return update

        update["status"] = QueryStatus.ANSWERED
        update["citations"] = citations
        return update

    @staticmethod
    def _route_after_generation(
        state: RagState,
    ) -> Literal["insufficient_evidence", "validate_citations"]:
        if state.get("status") == QueryStatus.INSUFFICIENT_EVIDENCE:
            return "insufficient_evidence"
        return "validate_citations"

    def _validate_citations(self, state: RagState) -> RagState:
        citations = state.get("citations", [])
        if not citations:
            raise ValueError("answered response requires at least one citation")
        return {"graph_path": self._append_path(state, "validate_citations")}

    def _clarification_required(self, state: RagState) -> RagState:
        return {
            "status": QueryStatus.CLARIFICATION_REQUIRED,
            "answer": None,
            "citations": [],
            "graph_path": self._append_path(state, "clarification_required"),
        }

    def _safety_blocked(self, state: RagState) -> RagState:
        return {
            "status": QueryStatus.SAFETY_BLOCKED,
            "answer": state["safety_blocked_message"],
            "citations": [],
            "graph_path": self._append_path(state, "safety_blocked"),
        }

    def _insufficient_evidence(self, state: RagState) -> RagState:
        return {
            "status": QueryStatus.INSUFFICIENT_EVIDENCE,
            "answer": state["insufficient_evidence_message"],
            "citations": [],
            "graph_path": self._append_path(state, "insufficient_evidence"),
        }
