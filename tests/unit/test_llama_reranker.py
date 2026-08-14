from __future__ import annotations

import pytest

from rag_ops_guard.adapters.reranking import llamacpp_reranker
from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_qwen_reranker_grades_each_document(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.03},
                    {"index": 0, "relevance_score": 0.97},
                ]
            }
        )

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    grades = adapter.grade("Calypso retries", ["Calypso policy", "Kafka policy"])

    assert [grade.relevant for grade in grades] == [True, False]
    assert [grade.score for grade in grades] == [0.97, 0.03]
    assert captured["url"] == "http://localhost:8082/v1/rerank"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen3-reranker-0.6b"
    assert body["documents"] == ["Calypso policy", "Kafka policy"]
    assert body["top_n"] == 2
    assert "Query: Calypso retries" in str(body["query"])
    assert "different target is not relevant" in str(body["query"])


def test_qwen_reranker_uses_native_yes_no_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.4999},
                ]
            }
        )

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    grades = adapter.grade("query", ["yes-doc", "no-doc"])

    assert grades[0].relevant is True
    assert grades[1].relevant is False


def test_qwen_reranker_rejects_invalid_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"results": [{"index": 0, "relevance_score": 1.2}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    with pytest.raises(ValueError, match="between 0 and 1"):
        adapter.grade("query", ["document"])


def test_qwen_reranker_rejects_incomplete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"results": [{"index": 0, "relevance_score": 0.8}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    with pytest.raises(ValueError, match="did not grade every document"):
        adapter.grade("query", ["doc-a", "doc-b"])
