from __future__ import annotations

import math

import httpx


class LlamaCppRerankerAdapter:
    """HTTP adapter for llama.cpp's /v1/rerank endpoint."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 30.0) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/rerank"
        self._model = model
        self._timeout_seconds = timeout_seconds

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        response = httpx.post(
            self._url,
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("reranker response must contain a results list")

        scores: list[float | None] = [None] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("reranker result entries must be objects")
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                raise ValueError("reranker result contains an invalid document index")

            if "relevance_score" in item:
                raw = item["relevance_score"]
                normalized = _normalized_score(raw)
            elif "score" in item:
                raw = item["score"]
                normalized = _sigmoid_score(raw)
            else:
                raise ValueError("reranker result is missing relevance_score")
            scores[index] = normalized

        if any(score is None for score in scores):
            raise ValueError("reranker response did not score every document")
        return [score for score in scores if score is not None]


def _normalized_score(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("reranker score must be numeric")
    score = float(value)
    if 0.0 <= score <= 1.0:
        return score
    return _sigmoid_score(score)


def _sigmoid_score(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("reranker score must be numeric")
    number = float(value)
    if number >= 0:
        factor = math.exp(-number)
        return 1.0 / (1.0 + factor)
    factor = math.exp(number)
    return factor / (1.0 + factor)
