from __future__ import annotations

import math
import re
from collections import Counter

from rag_ops_guard.domain.models import Evidence
from rag_ops_guard.retrieval.text import retrieval_text

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


class BM25Index:
    """Small in-memory BM25 index for the operational chunk corpus."""

    def __init__(self, evidence: list[Evidence], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._evidence = evidence
        self._k1 = k1
        self._b = b
        self._tokens = [_tokens(retrieval_text(item.chunk)) for item in evidence]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_length = sum(self._lengths) / max(1, len(self._lengths))
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )

    def search(self, query: str, *, limit: int = 20) -> list[Evidence]:
        query_tokens = _tokens(query)
        if not query_tokens or not self._evidence:
            return []

        ranked: list[tuple[float, int]] = []
        for index, counts in enumerate(self._term_counts):
            score = sum(self._term_score(term, index, counts[term]) for term in query_tokens)
            if score > 0:
                ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            Evidence(chunk=self._evidence[index].chunk, distance=None)
            for _, index in ranked[:limit]
        ]

    def _term_score(self, term: str, index: int, frequency: int) -> float:
        if frequency == 0:
            return 0.0
        total = len(self._evidence)
        document_frequency = self._document_frequency[term]
        idf = math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
        length = self._lengths[index]
        denominator = frequency + self._k1 * (
            1.0 - self._b + self._b * length / max(self._avg_length, 1.0)
        )
        return idf * (frequency * (self._k1 + 1.0)) / denominator


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]
