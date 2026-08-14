from __future__ import annotations

import math

import pytest

from rag_ops_guard.adapters.reranking import llamacpp_reranker
from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_reranker_restores_original_document_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.12},
                ]
            }
        )

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "bge-reranker-v2-m3")

    scores = adapter.score("Calypso retries", ["first", "second"])

    assert scores == [0.12, 0.93]
    assert captured["url"] == "http://localhost:8082/v1/rerank"
    assert captured["json"] == {
        "model": "bge-reranker-v2-m3",
        "query": "Calypso retries",
        "documents": ["first", "second"],
        "top_n": 2,
    }


def test_reranker_normalizes_raw_logit_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"results": [{"index": 0, "score": 0.0}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "reranker")

    assert adapter.score("query", ["doc"])[0] == pytest.approx(0.5)
    assert math.isfinite(adapter.score("query", ["doc"])[0])


def test_reranker_rejects_incomplete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"results": [{"index": 0, "relevance_score": 0.8}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "reranker")

    with pytest.raises(ValueError, match="did not score every document"):
        adapter.score("query", ["doc-a", "doc-b"])
