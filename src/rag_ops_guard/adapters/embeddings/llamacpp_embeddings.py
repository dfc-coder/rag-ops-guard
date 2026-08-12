from __future__ import annotations

import math

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class LlamaCppEmbeddingAdapter:
    def __init__(self, base_url: str, model: str, dimension: int = 1024) -> None:
        self._dimension = dimension
        self._client = OpenAIEmbeddings(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._normalize(vector) for vector in self._client.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._normalize(self._client.embed_query(text))

    def _normalize(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dimension:
            raise ValueError(f"expected {self._dimension} embedding dimensions, got {len(vector)}")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("zero embedding cannot be normalized")
        return [float(value / norm) for value in vector]
