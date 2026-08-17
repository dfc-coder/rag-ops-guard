from __future__ import annotations

from typing import Any

import pytest

from rag_ops_guard.adapters.embeddings import llamacpp_embeddings
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm import tokenizer
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter


class FakeEmbeddings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [3.0, 4.0]


class FakeHttpResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"tokens": [1, 2, 3]}


def test_embedding_adapter_normalizes_and_validates_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", FakeEmbeddings)
    adapter = LlamaCppEmbeddingAdapter("http://localhost:8081/v1", "embed", dimension=2)
    assert adapter.embed_query("hello") == pytest.approx([0.6, 0.8])
    assert adapter.embed_documents(["a", "b"])[1] == pytest.approx([0.6, 0.8])


def test_embedding_adapter_bounds_request_timeout_and_disables_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow/unreachable embedding backend must fail fast and deterministically."""
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", FakeEmbeddings)
    adapter = LlamaCppEmbeddingAdapter(
        "http://localhost:8081/v1",
        "embed",
        dimension=2,
        timeout_seconds=12.5,
    )

    assert adapter._client.kwargs["timeout"] == 12.5
    assert adapter._client.kwargs["max_retries"] == 0


def test_token_counter_calls_llama_tokenize(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeHttpResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeHttpResponse()

    monkeypatch.setattr(tokenizer.httpx, "post", fake_post)
    counter = LlamaCppTokenCounter("http://localhost:8081/v1")
    assert counter("hello") == 3
    assert captured["url"] == "http://localhost:8081/tokenize"
