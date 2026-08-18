from __future__ import annotations

import math
from collections.abc import Callable
from time import sleep
from typing import TypeVar

from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError
from pydantic import SecretStr

T = TypeVar("T")
_TRANSIENT_RETRY_DELAYS_SECONDS = (0.10, 0.25)


class LlamaCppEmbeddingAdapter:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int = 1024,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._dimension = dimension
        self._client = OpenAIEmbeddings(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
            timeout=timeout_seconds,
            max_retries=0,
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._with_transient_connection_retry(lambda: self._client.embed_documents(texts))
        return [self._normalize(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._with_transient_connection_retry(lambda: self._client.embed_query(text))
        return self._normalize(vector)

    def _with_transient_connection_retry(self, operation: Callable[[], T]) -> T:
        for delay in (*_TRANSIENT_RETRY_DELAYS_SECONDS, None):
            try:
                return operation()
            except APIConnectionError:
                if delay is None:
                    raise
                sleep(delay)
        raise AssertionError("unreachable")

    def _normalize(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dimension:
            raise ValueError(f"expected {self._dimension} embedding dimensions, got {len(vector)}")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("zero embedding cannot be normalized")
        return [float(value / norm) for value in vector]
