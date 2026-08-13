from __future__ import annotations

from time import perf_counter

from openai import APITimeoutError
from pydantic import ValidationError

from rag_ops_guard.domain.models import QueryRequest, QueryResponse, QueryStatus
from rag_ops_guard.graph.state import RagState
from rag_ops_guard.graph.workflow import RagWorkflow
from rag_ops_guard.observability.runtime import QUERY_LOGGER
from rag_ops_guard.retrieval.query_instruction import embedding_query


class TimedRagWorkflow(RagWorkflow):
    @staticmethod
    def _timings(state: RagState, **updates: float) -> dict[str, float]:
        return {**state.get("timings_ms", {}), **updates}

    def invoke(self, request: QueryRequest) -> QueryResponse:
        started = perf_counter()
        state: RagState = {
            "request_id": self._new_request_id(),
            "question": request.question.strip(),
            "context": request.context,
            "graph_path": [],
            "citations": [],
            "timings_ms": {},
        }
        result = self._graph.invoke(state)
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

    @staticmethod
    def _new_request_id() -> str:
        from uuid import uuid4

        return str(uuid4())

    def _analyze_query(self, state: RagState) -> RagState:
        started = perf_counter()
        update = super()._analyze_query(state)
        update["timings_ms"] = self._timings(
            state,
            analysis=round((perf_counter() - started) * 1000, 2),
        )
        return update

    def _retrieve_evidence(self, state: RagState) -> RagState:
        started = perf_counter()
        vector = self._embeddings.embed_query(embedding_query(state["normalized_question"]))
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
        update = super()._resolve_evidence(state)
        update["timings_ms"] = self._timings(
            state,
            resolver=round((perf_counter() - started) * 1000, 2),
        )
        return update

    def _generate_grounded_answer(self, state: RagState) -> RagState:
        started = perf_counter()
        try:
            update = super()._generate_grounded_answer(state)
        except (APITimeoutError, ValidationError) as exc:
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
                "answer": state.get("insufficient_evidence_message")
                or "The available documentation was not enough to answer.",
                "citations": [],
                "timings_ms": self._timings(state, generation=generation_ms),
                "graph_path": self._append_path(state, "generate_grounded_answer"),
            }

        update["timings_ms"] = self._timings(
            state,
            generation=round((perf_counter() - started) * 1000, 2),
        )
        return update
