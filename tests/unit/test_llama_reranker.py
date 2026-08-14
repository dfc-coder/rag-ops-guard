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


def _completion(yes: float | None, no: float | None) -> dict[str, object]:
    top: list[dict[str, object]] = []
    if yes is not None:
        top.append({"token": "yes", "logprob": yes})
    if no is not None:
        top.append({"token": "no", "logprob": no})
    return {"probs": [{"top_logprobs": top}]}


def test_qwen_reranker_grades_each_document(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            [
                _completion(yes=-0.05, no=-4.0),
                _completion(yes=-5.0, no=-0.02),
            ]
        )

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    grades = adapter.grade("Calypso retries", ["Calypso policy", "Kafka policy"])

    assert [grade.relevant for grade in grades] == [True, False]
    assert grades[0].score > 0.9
    assert grades[1].score < 0.1
    assert captured["url"] == "http://localhost:8082/completion"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["n_predict"] == 1
    assert body["temperature"] == -1.0
    prompts = body["prompt"]
    assert isinstance(prompts, list)
    assert "<Query>: Calypso retries" in prompts[0]
    assert "<Document>: Calypso policy" in prompts[0]
    assert "different target is not relevant" in prompts[0]


def test_qwen_reranker_uses_reference_missing_class_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse(_completion(yes=-0.01, no=None))

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    grade = adapter.grade("query", ["document"])[0]

    assert grade.relevant is True
    assert grade.score > 0.99


def test_qwen_reranker_rejects_response_without_yes_no_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse({"probs": [{"top_logprobs": [{"token": "maybe", "logprob": 0.0}]}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    with pytest.raises(ValueError, match="yes/no logit"):
        adapter.grade("query", ["document"])


def test_qwen_reranker_rejects_incomplete_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        del url, kwargs
        return FakeResponse(_completion(yes=-0.1, no=-3.0))

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter("http://localhost:8082", "qwen3-reranker-0.6b")

    with pytest.raises(ValueError, match="did not grade every document"):
        adapter.grade("query", ["doc-a", "doc-b"])
