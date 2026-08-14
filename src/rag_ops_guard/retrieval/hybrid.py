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
    "about",
    "this",
    "that",
    "does",
    "with",
    "from",
    "tell",
    "me",
    "para",
    "por",
    "que",
    "qué",
    "cual",
    "cuál",
    "como",
    "cómo",
    "este",
    "esta",
    "esto",
    "del",
    "las",
    "los",
    "una",
    "uno",
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
    """Dense + BM25 + RRF + deterministic resolver + cross-encoder reranking."""

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
        min_reranker_score: float = 0.5,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._objects = objects
        self._resolver = resolver
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._context_k = context_k
        self._min_reranker_score = min_reranker_score
        self._bm25: BM25Index | None = None

    def search(
        self,
        query: str,
        context: QueryContext,
        *,
        query_mode: QueryMode = "knowledge",
    ) -> KnowledgeSearchResult:
        dense_query = embedding_query(query) if query_mode == "knowledge" else query.strip()
        dense_vector = self._embeddings.embed_query(dense_query)
        dense = self._vectors.query(dense_vector, self._candidate_k)
        lexical = self._lexical_index().search(query, limit=self._candidate_k)
        fused = reciprocal_rank_fusion(dense=dense, lexical=lexical)

        fused_rank = {item.chunk.id: rank for rank, item in enumerate(fused)}
        resolved = self._resolver.resolve(fused, context, limit=max(1, len(fused)))
        resolved.sort(key=lambda item: fused_rank.get(item.chunk.id, len(fused_rank)))
        candidates = resolved[: self._candidate_k]

        scores = self._reranker.score(query, [_reranker_document(item) for item in candidates])
        if len(scores) != len(candidates):
            raise ValueError("reranker returned a score count that does not match candidates")

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], fused_rank.get(pair[0].chunk.id, len(fused_rank))),
        )
        admitted_pairs = [pair for pair in ranked if pair[1] >= self._min_reranker_score][
            : self._context_k
        ]
        admitted = [item for item, _score in admitted_pairs]
        top_score = ranked[0][1] if ranked else 0.0
        reranker_scores = {item.chunk.id: round(score, 6) for item, score in ranked}

        return KnowledgeSearchResult(
            dense=dense,
            lexical=lexical,
            fused=fused,
            admitted=admitted,
            relevance=round(top_score, 6),
            lexical_relevance=retrieval_lexical_relevance(query, admitted=admitted),
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
    """Legacy diagnostic score; production admission is decided by the cross-encoder."""
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
        f"Document type: {chunk.metadata.document_type.value}",
    ]
    if section:
        parts.append(f"Section: {section}")
    parts.append(chunk.text)
    return "\n".join(parts)


def _informative_tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS
    }
