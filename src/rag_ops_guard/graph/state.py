from __future__ import annotations

from typing import TypedDict

from rag_ops_guard.domain.models import Citation, Evidence, QueryContext, QueryStatus


class RagState(TypedDict, total=False):
    request_id: str
    question: str
    normalized_question: str
    context: QueryContext
    status: QueryStatus
    clarification_question: str | None
    safety_blocked_message: str | None
    insufficient_evidence_message: str | None
    retrieved_evidence: list[Evidence]
    resolved_evidence: list[Evidence]
    answer: str | None
    citations: list[Citation]
    graph_path: list[str]
    safety_category: str
