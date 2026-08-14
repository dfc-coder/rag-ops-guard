from __future__ import annotations

import httpx

from rag_ops_guard.ports import RerankGrade

_DEFAULT_INSTRUCTION = (
    "Determine whether each document directly answers the enterprise integration-operations "
    "query. Respect every explicit constraint in the query, including system, API, product, "
    "protocol, environment, version, and requested operation. A document about a different "
    "target is not relevant merely because it describes a similar operation."
)


class LlamaCppRerankerAdapter:
    """Qwen3 reranker adapter backed by llama.cpp's native yes/no classifier."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/rerank"
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._instruction = instruction

    def grade(self, query: str, documents: list[str]) -> list[RerankGrade]:
        if not documents:
            return []

        response = httpx.post(
            self._url,
            json={
                "model": self._model,
                "query": f"Instruct: {self._instruction}\nQuery: {query.strip()}",
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

        # Qwen3-Reranker GGUF exposes classifier.output_labels=[yes,no]. The returned
        # relevance_score is the model-native yes probability; 0.5 is therefore the
        # yes-vs-no decision boundary, not a corpus-tuned admission threshold.
        return [
            RerankGrade(relevant=score >= 0.5, score=score) for score in scores if score is not None
        ]
