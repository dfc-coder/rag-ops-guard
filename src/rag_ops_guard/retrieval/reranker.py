from __future__ import annotations

import math
import re
from collections import Counter

from rag_ops_guard.domain.models import Evidence

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def rerank_evidence(question: str, evidence: list[Evidence]) -> list[Evidence]:
    """Fuse semantic rank with lexical overlap without domain-specific rules."""
    if len(evidence) < 2:
        return evidence

    query_tokens = set(_tokens(question))
    if not query_tokens:
        return evidence

    document_tokens = [
        set(_tokens(f"{item.chunk.title}\n{item.chunk.text}")) for item in evidence
    ]
    document_frequency = Counter(
        token for tokens in document_tokens for token in query_tokens.intersection(tokens)
    )
    total = len(evidence)

    lexical_scores: list[float] = []
    for tokens in document_tokens:
        score = 0.0
        for token in query_tokens.intersection(tokens):
            score += math.log1p((total + 1) / (document_frequency[token] + 0.5))
        lexical_scores.append(score)

    semantic_order = sorted(
        range(total),
        key=lambda index: (
            evidence[index].distance is None,
            evidence[index].distance or 0.0,
        ),
    )
    semantic_rank = {index: rank for rank, index in enumerate(semantic_order, start=1)}

    lexical_order = sorted(
        range(total),
        key=lambda index: (-lexical_scores[index], semantic_rank[index]),
    )
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}

    def fused_score(index: int) -> float:
        semantic = 1.0 / (60 + semantic_rank[index])
        lexical = 0.0
        if lexical_scores[index] > 0:
            lexical = 1.0 / (60 + lexical_rank[index])
        return semantic + lexical

    ranked = sorted(range(total), key=fused_score, reverse=True)
    return [evidence[index] for index in ranked]


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]
