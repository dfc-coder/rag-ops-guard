from __future__ import annotations

import httpx

from rag_ops_guard.ports import RerankGrade

_DEFAULT_INSTRUCTION = (
    "Determine whether each document directly answers the enterprise integration-operations "
    "query. Respect every explicit constraint in the query, including system, API, product, "
    "protocol, environment, version, and requested operation. A document about a different "
    "target is not relevant merely because it describes a similar operation."
)
_DEFAULT_BATCH_SIZE = 8


class LlamaCppRerankerAdapter:
    """HTTP reranker adapter compatible with llama.cpp v1 and OpenVINO Model Server v3."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch_size must be at least 1")
        normalized = base_url.rstrip("/")
        self._openvino = normalized.endswith("/v3")
        self._url = (
            f"{normalized}/rerank"
            if normalized.endswith(("/v1", "/v3"))
            else f"{normalized}/v1/rerank"
        )
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._instruction = instruction
        self._batch_size = batch_size

    def _query(self, query: str) -> str:
        # OVMS prepares the Qwen sequence-classification reranker with its own task template.
        # llama.cpp's native classifier still benefits from the explicit relevance instruction.
        if self._openvino:
            return query.strip()
        return f"Instruct: {self._instruction}\nQuery: {query.strip()}"

    def _grade_batch(self, query: str, documents: list[str]) -> list[float]:
        response = httpx.post(
            self._url,
            json={
                "model": self._model,
                "query": self._query(query),
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
            score = item.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                raise ValueError("reranker result contains an invalid document index")
            if not isinstance(score, int | float):
                raise ValueError("reranker result is missing a numeric relevance_score")
            normalized = float(score)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError("reranker relevance_score must be between 0 and 1")
            scores[index] = normalized

        if any(score is None for score in scores):
            raise ValueError("reranker response did not grade every document")
        return [score for score in scores if score is not None]

    def grade(self, query: str, documents: list[str]) -> list[RerankGrade]:
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            scores.extend(self._grade_batch(query, batch))

        return [RerankGrade(relevant=score >= 0.5, score=score) for score in scores]
