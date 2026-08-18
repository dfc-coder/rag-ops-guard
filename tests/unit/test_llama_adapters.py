from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIConnectionError

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


class TransientEmbeddings(FakeEmbeddings):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.query_calls = 0
        self.document_calls = 0

    @staticmethod
    def _connection_error() -> APIConnectionError:
        return APIConnectionError(request=httpx.Request("POST", "http://ovms/v3/embeddings"))

    def embed_query(self, text: str) -> list[float]:
        del text
        self.query_calls += 1
        if self.query_calls == 1:
            raise self._connection_error()
        return [3.0, 4.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.document_calls == 1:
            raise self._connection_error()
        return [[3.0, 4.0] for _ in texts]


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


def test_embedding_adapter_bounds_request_timeout_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow/unreachable compatible backend must fail fast and receive raw text."""
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", FakeEmbeddings)
    adapter = LlamaCppEmbeddingAdapter(
        "http://localhost:8081/v1",
        "embed",
        dimension=2,
        timeout_seconds=12.5,
    )

    assert adapter._client.kwargs["timeout"] == 12.5
    assert adapter._client.kwargs["max_retries"] == 0
    assert adapter._client.kwargs["check_embedding_ctx_length"] is False


def test_embedding_adapter_retries_one_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llamacpp_embeddings, "OpenAIEmbeddings", TransientEmbeddings)
    monkeypatch.setattr(llamacpp_embeddings, "sleep", lambda _seconds: None, raising=False)
    adapter = LlamaCppEmbeddingAdapter("http://localhost:8081/v1", "embed", dimension=2)

    assert adapter.embed_query("hello") == pytest.approx([0.6, 0.8])
    assert adapter.embed_documents(["a"])[0] == pytest.approx([0.6, 0.8])
    assert adapter._client.query_calls == 2
    assert adapter._client.document_calls == 2


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
