from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from openai import APITimeoutError, LengthFinishReasonError
from pydantic import ValidationError

from rag_ops_guard.domain.errors import CitationValidationError, EvidenceConflictError
from rag_ops_guard.domain.models import Evidence, QueryRequest, QueryResponse, QueryStatus
from rag_ops_guard.graph.prompts import answer_prompt
from rag_ops_guard.graph.state import RagState
from rag_ops_guard.graph.workflow import RagWorkflow
from rag_ops_guard.observability.runtime import QUERY_LOGGER
from rag_ops_guard.retrieval.citations import validate_citations
from rag_ops_guard.retrieval.query_instruction import embedding_query
from rag_ops_guard.retrieval.reranker import rerank_evidence


class TimedRagWorkflow(RagWorkflow):
    """Production hot path: retrieve, resolve, then make exactly one LLM call."""

    @staticmethod
    def _timings(state: RagState, **updates: float) -> dict[str, float]:
        return {**state.get("timings_ms", {}), **updates}

    def _build_graph(self) -> Any:
        builder = StateGraph(RagState)
        builder.add_node("validate_request", self._validate_request)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("resolve_evidence", self._resolve_evidence)
        builder.add_node("generate_grounded_answer", self._generate_grounded_answer)

        builder.add_edge(START, "validate_request")
        builder.add_edge("validate_request", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "resolve_evidence")
        builder.add_edge("resolve_evidence", "generate_grounded_answer")
        builder.add_edge("generate_grounded_answer", END)
        return builder.compile()

    def invoke(self, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        request_id = str(uuid4())
        state: RagState = {
            "request_id": request_id,
            "question": request.question.strip(),
            "context": request.context,
            "graph_path": [],
            "citations": [],
            "timings_ms": {},
        }
        metadata = {
            "request_id": request_id,
            "system": request.context.system or "",
            "environment": request.context.environment or "",
            "api_version": request.context.api_version or "",
        }
        result = self._graph.invoke(
            state,
            config={
                "run_name": "rag-query",
                "tags": ["rag-ops-guard", "one-pass"],
                "metadata": metadata,
            },
        )
        timings = dict(result.get("timings_ms", {}))
        timings["total"] = round((perf_counter() - started) * 1000, 2)
        return QueryResponse(
            request_id=result["request_id"],
            status=result["status"],
            answer=result.get("answer"),
            clarification_question=result.get("clarification_question"),
            citations=result.get("citations", []),
            timings_ms=timings,
        )

    def _retrieve_evidence(self, state: RagState) -> RagState:
        started = perf_counter()
        vector = self._embeddings.embed_query(embedding_query(state["question"]))
        embedding_ms = round((perf_counter() - started) * 1000, 2)

        started = perf_counter()
        evidence = self._vectors.query(vector, self._top_k)
        retrieval_ms = round((perf_counter() - started) * 1000, 2)

        QUERY_LOGGER.info(
            "retrieval_completed",
            extra={
                "request_id": state["request_id"],
                "retrieved_documents": len(evidence),
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval_ms,
            },
        )
        return {
            "retrieved_evidence": evidence,
            "timings_ms": self._timings(
                state,
                embedding=embedding_ms,
                retrieval=retrieval_ms,
            ),
            "graph_path": self._append_path(state, "retrieve_evidence"),
        }

    def _resolve_evidence(self, state: RagState) -> RagState:
        started = perf_counter()
        try:
            candidates = self._resolver.resolve(
                state.get("retrieved_evidence", []),
                state["context"],
                limit=self._top_k,
            )
            resolved = rerank_evidence(state["question"], candidates)[: self._context_k]
        except EvidenceConflictError as exc:
            QUERY_LOGGER.warning(
                "evidence_conflict",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            resolved = []

        resolver_ms = round((perf_counter() - started) * 1000, 2)
        QUERY_LOGGER.info(
            "evidence_resolved",
            extra={
                "request_id": state["request_id"],
                "resolved_documents": len(resolved),
                "resolved_titles": [item.chunk.title for item in resolved],
                "resolved_distances": [item.distance for item in resolved],
            },
        )
        return {
            "resolved_evidence": resolved,
            "timings_ms": self._timings(state, resolver=resolver_ms),
            "graph_path": self._append_path(state, "resolve_evidence"),
        }

    def _generate_grounded_answer(self, state: RagState) -> RagState:
        started = perf_counter()
        try:
            grounded = self._chat.generate_answer(
                answer_prompt(state["question"], state.get("resolved_evidence", []))
            )
        except (APITimeoutError, LengthFinishReasonError, ValidationError) as exc:
            generation_ms = round((perf_counter() - started) * 1000, 2)
            QUERY_LOGGER.warning(
                "generation_failed",
                extra={
                    "request_id": state["request_id"],
                    "generation_ms": generation_ms,
                    "reason": type(exc).__name__,
                },
            )
            return {
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": None,
                "clarification_question": None,
                "citations": [],
                "timings_ms": self._timings(state, generation=generation_ms),
                "graph_path": self._append_path(state, "generate_grounded_answer"),
            }

        generation_ms = round((perf_counter() - started) * 1000, 2)
        common: RagState = {
            "timings_ms": self._timings(state, generation=generation_ms),
            "graph_path": self._append_path(state, "generate_grounded_answer"),
        }

        if grounded.status == "clarification_required":
            return {
                **common,
                "status": QueryStatus.CLARIFICATION_REQUIRED,
                "answer": None,
                "clarification_question": grounded.answer,
                "citations": [],
            }

        if grounded.status == "safety_blocked":
            return {
                **common,
                "status": QueryStatus.SAFETY_BLOCKED,
                "answer": grounded.answer,
                "clarification_question": None,
                "citations": [],
            }

        if grounded.status == "insufficient_evidence":
            return {
                **common,
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": grounded.answer,
                "clarification_question": None,
                "citations": [],
            }

        try:
            citation_ids = self._expand_citation_refs(
                grounded.citation_ids,
                state.get("resolved_evidence", []),
            )
            citations = validate_citations(
                citation_ids,
                state.get("resolved_evidence", []),
            )
        except CitationValidationError as exc:
            QUERY_LOGGER.warning(
                "invalid_generated_citation",
                extra={"request_id": state["request_id"], "detail": str(exc)},
            )
            return {
                **common,
                "status": QueryStatus.INSUFFICIENT_EVIDENCE,
                "answer": None,
                "clarification_question": None,
                "citations": [],
            }

        return {
            **common,
            "status": QueryStatus.ANSWERED,
            "answer": grounded.answer,
            "clarification_question": None,
            "citations": citations,
        }

    @staticmethod
    def _expand_citation_refs(
        citation_ids: list[str], evidence: list[Evidence]
    ) -> list[str]:
        lookup = {
            f"E{index}": item.chunk.id
            for index, item in enumerate(evidence, start=1)
        }
        lookup.update({item.chunk.id: item.chunk.id for item in evidence})
        return [lookup.get(citation_id, citation_id) for citation_id in citation_ids]
