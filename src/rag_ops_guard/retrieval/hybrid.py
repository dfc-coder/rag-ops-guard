from __future__ import annotations

import re
from dataclasses import dataclass

from rag_ops_guard.domain.models import Chunk, Evidence, QueryContext
from rag_ops_guard.ports import EmbeddingProvider, ObjectStore, VectorStore
from rag_ops_guard.retrieval.bm25 import BM25Index
from rag_ops_guard.retrieval.query_instruction import embedding_query
from rag_ops_guard.retrieval.reranker import rerank_evidence
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
}


@dataclass(frozen=True)
class KnowledgeSearchResult:
    dense: list[Evidence]
    lexical: list[Evidence]
    fused: list[Evidence]
    admitted: list[Evidence]
    relevance: float = 0.0


class KnowledgeSearch:
    """Single production retrieval path: dense + BM25 + RRF + policy resolver + rerank."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        objects: ObjectStore,
        resolver: EvidenceResolver,
        candidate_k: int = 20,
        context_k: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._objects = objects
        self._resolver = resolver
        self._candidate_k = candidate_k
        self._context_k = context_k
        self._bm25: BM25Index | None = None

    def search(self, query: str, context: QueryContext) -> KnowledgeSearchResult:
        dense_vector = self._embeddings.embed_query(embedding_query(query))
        dense = self._vectors.query(dense_vector, self._candidate_k)
        lexical = self._lexical_index().search(query, limit=self._candidate_k)
        fused = reciprocal_rank_fusion(dense=dense, lexical=lexical)

        fused_rank = {item.chunk.id: rank for rank, item in enumerate(fused)}
        resolved = self._resolver.resolve(fused, context, limit=max(1, len(fused)))
        resolved.sort(key=lambda item: fused_rank.get(item.chunk.id, len(fused_rank)))
        candidates = resolved[: self._candidate_k]
        admitted = rerank_evidence(query, candidates)[: self._context_k]

        return KnowledgeSearchResult(
            dense=dense,
            lexical=lexical,
            fused=fused,
            admitted=admitted,
            relevance=retrieval_relevance(query, dense, admitted),
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


def retrieval_relevance(query: str, dense: list[Evidence], admitted: list[Evidence]) -> float:
    semantic = 0.0
    distances = [item.distance for item in dense if item.distance is not None]
    if distances:
        semantic = max(0.0, min(1.0, 1.0 - min(distances)))

    query_tokens = _informative_tokens(query)
    lexical = 0.0
    if query_tokens:
        for item in admitted:
            document_tokens = _informative_tokens(f"{item.chunk.title}\n{item.chunk.text}")
            lexical = max(
                lexical, len(query_tokens.intersection(document_tokens)) / len(query_tokens)
            )

    return round(max(semantic, lexical), 6)


def _informative_tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS
    }
