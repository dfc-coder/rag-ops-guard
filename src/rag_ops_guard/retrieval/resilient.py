from __future__ import annotations

import logging

from rag_ops_guard.domain.models import Evidence, QueryContext
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch, KnowledgeSearchResult, QueryMode

logger = logging.getLogger(__name__)


class ResilientKnowledgeSearch(KnowledgeSearch):
    """Knowledge search with observable stages and a fail-safe recall fallback.

    The normal path keeps the instruction-wrapped Qwen embedding query. If that path finds no
    admissible evidence, the exact same query is retried once with the raw query embedding. The
    fallback does not lower reranker thresholds, bypass the resolver, or admit unsupported facts;
    it only gives short/multilingual/entity-heavy queries a second recall path before abstaining.
    """

    def search(
        self,
        query: str,
        context: QueryContext,
        *,
        query_mode: QueryMode = "knowledge",
        ranking_query: str | None = None,
    ) -> KnowledgeSearchResult:
        primary = super().search(
            query,
            context,
            query_mode=query_mode,
            ranking_query=ranking_query,
        )
        _log_attempt("primary", query_mode, query, context, primary)

        if query_mode != "knowledge" or primary.supported:
            return primary

        fallback = super().search(
            query,
            context,
            query_mode="probe",
            ranking_query=ranking_query,
        )
        _log_attempt("raw-query-fallback", "probe", query, context, fallback)
        if fallback.supported:
            logger.info(
                "knowledge-search recovered with raw-query fallback query=%r environment=%r",
                query,
                context.environment,
            )
            return fallback
        return primary


def _evidence_names(items: list[Evidence], *, limit: int = 8) -> list[str]:
    return [
        f"{item.chunk.logical_id}#{item.chunk.chunk_index}"
        for item in items[:limit]
    ]


def _log_attempt(
    attempt: str,
    mode: QueryMode,
    query: str,
    context: QueryContext,
    result: KnowledgeSearchResult,
) -> None:
    logger.info(
        "knowledge-search attempt=%s mode=%s query=%r system=%r environment=%r api_version=%r "
        "supported=%s relevance=%.4f dense=%s lexical=%s fused=%s reranker=%s admitted=%s",
        attempt,
        mode,
        query,
        context.system,
        context.environment,
        context.api_version,
        result.supported,
        result.relevance,
        _evidence_names(result.dense),
        _evidence_names(result.lexical),
        _evidence_names(result.fused),
        result.reranker_scores,
        _evidence_names(result.admitted),
    )
