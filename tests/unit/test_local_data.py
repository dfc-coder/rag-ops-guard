from __future__ import annotations

from typing import Any

import pytest

from scripts.local import ensure_data


class FakeVectorsClient:
    def __init__(self, vectors: list[dict[str, object]]) -> None:
        self._vectors = vectors
        self.kwargs: dict[str, object] = {}

    def query_vectors(self, **kwargs: Any) -> dict[str, object]:
        self.kwargs = kwargs
        return {"vectors": self._vectors}


def test_has_vectors_uses_query_vectors_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVectorsClient([{"key": "chunk-1"}])
    monkeypatch.setattr(ensure_data, "client", lambda service, **kwargs: fake)

    assert ensure_data.has_vectors() is True
    assert fake.kwargs["vectorBucketName"] == ensure_data.VECTOR_BUCKET
    assert fake.kwargs["indexName"] == ensure_data.VECTOR_INDEX
    assert fake.kwargs["topK"] == 1

    query_vector = fake.kwargs["queryVector"]
    assert isinstance(query_vector, dict)
    probe = query_vector["float32"]
    assert isinstance(probe, list)
    assert len(probe) == ensure_data.VECTOR_DIMENSION
    assert probe[0] == 1.0
    assert all(value == 0.0 for value in probe[1:])


def test_has_vectors_returns_false_for_empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVectorsClient([])
    monkeypatch.setattr(ensure_data, "client", lambda service, **kwargs: fake)

    assert ensure_data.has_vectors() is False
