from __future__ import annotations

import math

import httpx

from rag_ops_guard.ports import RerankGrade

_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".'
)
_DEFAULT_INSTRUCTION = (
    "Determine whether the document directly answers the enterprise integration-operations "
    "query. All explicit constraints in the query, including system, API, product, protocol, "
    "environment, version, and requested operation, must be satisfied. A document about a "
    "different target is not relevant merely because it describes a similar operation."
)
_MISSING_CLASS_LOGPROB = -10.0


class LlamaCppRerankerAdapter:
    """Qwen3 reranker adapter using its native yes/no relevance objective."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/completion"
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._instruction = instruction

    def grade(self, query: str, documents: list[str]) -> list[RerankGrade]:
        if not documents:
            return []

        prompts = [self._prompt(query, document) for document in documents]
        response = httpx.post(
            self._url,
            json={
                "prompt": prompts,
                "n_predict": 1,
                "temperature": -1.0,
                "n_probs": 32,
                "cache_prompt": True,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload if isinstance(payload, list) else [payload]
        if len(items) != len(documents):
            raise ValueError("reranker response did not grade every document")

        grades: list[RerankGrade] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("reranker completion entries must be objects")
            yes_logprob, no_logprob = _yes_no_logprobs(item)
            yes_probability = _binary_probability(yes_logprob, no_logprob)
            grades.append(
                RerankGrade(
                    relevant=yes_logprob > no_logprob,
                    score=yes_probability,
                )
            )
        return grades

    def _prompt(self, query: str, document: str) -> str:
        pair = (
            f"<Instruct>: {self._instruction}\n"
            f"<Query>: {query.strip()}\n"
            f"<Document>: {document.strip()}"
        )
        return (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{pair}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def _yes_no_logprobs(item: dict[str, object]) -> tuple[float, float]:
    probabilities = item.get("probs") or item.get("completion_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise ValueError("reranker completion is missing token probabilities")
    first = probabilities[0]
    if not isinstance(first, dict):
        raise ValueError("reranker probability entry must be an object")

    top = first.get("top_logprobs")
    if not isinstance(top, list):
        raise ValueError("reranker completion is missing top_logprobs")

    values: dict[str, float] = {}
    for candidate in top:
        if not isinstance(candidate, dict):
            continue
        token = candidate.get("token")
        logprob = candidate.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, int | float):
            continue
        normalized = token.strip().casefold()
        if normalized in {"yes", "no"}:
            values[normalized] = float(logprob)

    if not values:
        raise ValueError("reranker completion did not expose a yes/no logit")
    return (
        values.get("yes", _MISSING_CLASS_LOGPROB),
        values.get("no", _MISSING_CLASS_LOGPROB),
    )


def _binary_probability(yes_logprob: float, no_logprob: float) -> float:
    """Normalize the two class logits as in the Qwen reranker reference implementation."""
    delta = no_logprob - yes_logprob
    if delta >= 0:
        factor = math.exp(-delta)
        return factor / (1.0 + factor)
    factor = math.exp(delta)
    return 1.0 / (1.0 + factor)
