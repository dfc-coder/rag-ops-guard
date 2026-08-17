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


def test_openvino_qwen_seq_cls_reranker_uses_required_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse({"results": [{"index": 0, "relevance_score": 0.97}]})

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter(
        "http://localhost:8083/v3",
        "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov",
    )

    adapter.grade("Cuantos reintentos permite Calypso?", ["Payment Retry Policy body"])

    assert captured["url"] == "http://localhost:8083/v3/rerank"
    body = captured["json"]
    assert isinstance(body, dict)
    query = str(body["query"])
    documents = body["documents"]
    assert isinstance(documents, list)
    assert query.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert "<Instruct>:" in query
    assert "<Query>: Cuantos reintentos permite Calypso?" in query
    assert query.endswith("\n")
    assert documents == [
        "<Document>: Payment Retry Policy body<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]


def test_qwen_reranker_batches_documents_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches: list[list[str]] = []
    top_ns: list[int] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url == "http://localhost:8082/v1/rerank"
        body = kwargs["json"]
        assert isinstance(body, dict)
        documents = body["documents"]
        assert isinstance(documents, list)
        batches.append(documents)
        top_n = body["top_n"]
        assert isinstance(top_n, int)
        top_ns.append(top_n)
        return FakeResponse(
            {
                "results": [
                    {
                        "index": index,
                        "relevance_score": 0.9 if document.endswith("0") else 0.1,
                    }
                    for index, document in enumerate(documents)
                ]
            }
        )

    monkeypatch.setattr(llamacpp_reranker.httpx, "post", fake_post)
    adapter = LlamaCppRerankerAdapter(
        "http://localhost:8082",
        "qwen3-reranker-0.6b",
        batch_size=4,
    )
    documents = [f"doc-{index}" for index in range(10)]

    grades = adapter.grade("query", documents)

    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert [document for batch in batches for document in batch] == documents
    assert top_ns == [4, 4, 2]
    assert [grade.score for grade in grades] == [0.9] + [0.1] * 9
    assert [grade.relevant for grade in grades] == [True] + [False] * 9


def test_qwen_reranker_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        LlamaCppRerankerAdapter(
            "http://localhost:8082",
            "qwen3-reranker-0.6b",
            batch_size=0,
        )


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
