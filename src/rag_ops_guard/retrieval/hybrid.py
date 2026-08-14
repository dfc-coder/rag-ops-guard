from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rag_ops_guard.domain.models import Chunk, Evidence, QueryContext
from rag_ops_guard.ports import EmbeddingProvider, ObjectStore, Reranker, VectorStore
from rag_ops_guard.retrieval.bm25 import BM25Index
from rag_ops_guard.retrieval.query_instruction import embedding_query
from rag_ops_guard.retrieval.resolver import EvidenceResolver

_RRF_K = 60
_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)
_STOPWORDS = {
    "the",
    "and",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "about",
    "this",
    "that",
    "does",
    "with",
    "from",
    "tell",
    "me",
    "can",
    "could",
    "should",
    "would",
    "is",
    "are",
    "do",
    "did",
    "para",
    "por",
    "que",
    "qué",
    "cual",
    "cuál",
    "cuales",
    "cuáles",
    "como",
    "cómo",
    "cuando",
    "cuándo",
    "donde",
    "dónde",
    "quien",
    "quién",
    "quienes",
    "quiénes",
    "cuantos",
    "cuántos",
    "cuantas",
    "cuántas",
    "este",
    "esta",
    "esto",
    "del",
    "las",
    "los",
    "una",
    "uno",
    "se",
    "entonces",
    "con",
    "sobre",
    "pasa",
    "sucede",
    "hace",
    "sabes",
    "dime",
    "decime",
    "contame",
}
_GENERIC_OPERATION_TOKENS = {
    "api",
    "dlq",
    "http",
    "https",
    "id",
    "ids",
    "json",
    "p1",
    "p2",
    "p3",
    "p4",
    "rest",
    "sla",
    "sql",
    "xml",
}

QueryMode = Literal["knowledge", "probe"]


@dataclass(frozen=True)
class KnowledgeSearchResult:
    dense: list[Evidence]
    lexical: list[Evidence]
    fused: list[Evidence]
    admitted: list[Evidence]
    relevance: float = 0.0
    lexical_relevance: float = 0.0
    supported: bool = False
    reranker_scores: dict[str, float] = field(default_factory=dict)


class KnowledgeSearch:
    """Dense + BM25 + RRF + resolver + learned relevance grading."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        objects: ObjectStore,
        resolver: EvidenceResolver,
        reranker: Reranker,
        candidate_k: int = 20,
        context_k: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._objects = objects
        self._resolver = resolver
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._context_k = context_k
        self._bm25: BM25Index | None = None

    def search(
        self,
        query: str,
        context: QueryContext,
        *,
        query_mode: QueryMode = "knowledge",
        ranking_query: str | None = None,
    ) -> KnowledgeSearchResult:
        """Retrieve broadly, then grade candidates against the resolved user intent.

        For contextual follow-ups, ``query`` is the standalone rewrite used for recall and
        ``ranking_query`` is the literal current turn. The reranker receives both so it keeps
        the resolved topic without losing what the user actually asked in the follow-up.
        """
        dense_query = embedding_query(query) if query_mode == "knowledge" else query.strip()
        dense_vector = self._embeddings.embed_query(dense_query)
        dense = self._vectors.query(dense_vector, self._candidate_k)
        lexical = self._lexical_index().search(query, limit=self._candidate_k)
        fused = reciprocal_rank_fusion(dense=dense, lexical=lexical)

        fused_rank = {item.chunk.id: rank for rank, item in enumerate(fused)}
        resolved = self._resolver.resolve(fused, context, limit=max(1, len(fused)))
        resolved.sort(key=lambda item: fused_rank.get(item.chunk.id, len(fused_rank)))
        candidates = resolved[: self._candidate_k]

        standalone_query = query.strip()
        literal_query = (ranking_query or "").strip()
        relevance_query = (
            f"{standalone_query}\n{literal_query}"
            if literal_query and literal_query != standalone_query
            else standalone_query
        )
        if not _candidates_cover_explicit_anchors(relevance_query, candidates):
            return KnowledgeSearchResult(
                dense=dense,
                lexical=lexical,
                fused=fused,
                admitted=[],
                relevance=0.0,
                lexical_relevance=0.0,
                supported=False,
                reranker_scores={},
            )

        grades = self._reranker.grade(
            relevance_query,
            [_reranker_document(item) for item in candidates],
        )
        if len(grades) != len(candidates):
            raise ValueError("reranker returned a grade count that does not match candidates")

        ranked = sorted(
            zip(candidates, grades, strict=True),
            key=lambda pair: (-pair[1].score, fused_rank.get(pair[0].chunk.id, len(fused_rank))),
        )
        admitted_pairs = [pair for pair in ranked if pair[1].relevant][: self._context_k]
        admitted = [item for item, _grade in admitted_pairs]
        top_score = ranked[0][1].score if ranked else 0.0
        reranker_scores = {item.chunk.id: round(grade.score, 6) for item, grade in ranked}

        return KnowledgeSearchResult(
            dense=dense,
            lexical=lexical,
            fused=fused,
            admitted=admitted,
            relevance=round(top_score, 6),
            lexical_relevance=retrieval_lexical_relevance(relevance_query, admitted=admitted),
            supported=bool(admitted),
            reranker_scores=reranker_scores,
        )

    def refresh(self) -> None:
        self._bm25 = self._build_lexical_index()

    def _lexical_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = self._build_lexical_index()
        return self._bm25

    def _build_lexical_index(self) -> BM25Index:
        evidence: list[Evidence] = []
        for key in self._objects.list_keys("chunks/"):
            if not key.endswith(".json"):
                continue
            chunk = Chunk.model_validate_json(self._objects.get_text(key))
            evidence.append(Evidence(chunk=chunk, distance=None))
        return BM25Index(evidence)


def reciprocal_rank_fusion(
    *,
    dense: list[Evidence],
    lexical: list[Evidence],
    k: int = _RRF_K,
) -> list[Evidence]:
    scores: dict[str, float] = {}
    items: dict[str, Evidence] = {}

    for ranking in (dense, lexical):
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            current = items.get(chunk_id)
            if current is None or (current.distance is None and item.distance is not None):
                items[chunk_id] = item

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [items[chunk_id] for chunk_id in ordered_ids]


def retrieval_relevance(query: str, *, admitted: list[Evidence]) -> float:
    """Legacy diagnostic score; production admission is decided by learned grading."""
    semantic = 0.0
    distances = [item.distance for item in admitted if item.distance is not None]
    if distances:
        semantic = max(0.0, min(1.0, 1.0 - min(distances)))

    lexical = retrieval_lexical_relevance(query, admitted=admitted)
    if semantic > 0.0 and lexical > 0.0:
        score = 0.6 * semantic + 0.4 * lexical
    elif lexical > 0.0:
        score = 0.75 * lexical
    else:
        score = semantic
    return round(max(0.0, min(1.0, score)), 6)


def retrieval_lexical_relevance(query: str, *, admitted: list[Evidence]) -> float:
    """Diagnostic informative-token overlap against admitted evidence."""
    query_tokens = _informative_tokens(query)
    if not query_tokens:
        return 0.0

    lexical = 0.0
    for item in admitted:
        document_tokens = _informative_tokens(f"{item.chunk.title}\n{item.chunk.text}")
        lexical = max(lexical, len(query_tokens.intersection(document_tokens)) / len(query_tokens))
    return round(max(0.0, min(1.0, lexical)), 6)


def _reranker_document(item: Evidence) -> str:
    chunk = item.chunk
    section = " > ".join(chunk.header_path)
    parts = [
        f"Title: {chunk.title}",
        f"System: {chunk.metadata.system}",
        f"Environment: {chunk.metadata.environment}",
        f"Version: {chunk.version}",
        f"Document type: {chunk.metadata.document_type.value}",
    ]
    if section:
        parts.append(f"Section: {section}")
    parts.append(chunk.text)
    return "\n".join(parts)


def _candidates_cover_explicit_anchors(query: str, candidates: list[Evidence]) -> bool:
    anchors = _explicit_query_anchors(query)
    if not anchors:
        return True

    candidate_tokens: set[str] = set()
    for item in candidates:
        candidate_tokens.update(_all_tokens(_reranker_document(item)))
    return anchors.issubset(candidate_tokens)


def _explicit_query_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        folded = token.casefold()
        if folded in _STOPWORDS or folded in _GENERIC_OPERATION_TOKENS:
            continue
        has_letter = any(char.isalpha() for char in token)
        has_internal_upper = any(char.isupper() for char in token[1:])
        if has_letter and (token.isupper() or has_internal_upper or token[:1].isupper()):
            anchors.add(folded)
    return anchors


def _all_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def _informative_tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS
    }
