from __future__ import annotations

import io
import json
from typing import Any

import pytest

from scripts.local import ensure_data


class FakeS3Client:
    def __init__(self, manifests: dict[str, dict[str, object]]) -> None:
        self._manifests = manifests

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        if Key not in self._manifests:
            raise ensure_data.ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        payload = json.dumps(self._manifests[Key]).encode()
        return {"Body": io.BytesIO(payload)}


class FakeVectorsClient:
    def __init__(self, available: set[str]) -> None:
        self._available = available
        self.calls: list[list[str]] = []

    def get_vectors(self, **kwargs: Any) -> dict[str, object]:
        keys = list(kwargs["keys"])
        self.calls.append(keys)
        return {"vectors": [{"key": key} for key in keys if key in self._available]}


def _patch_clients(
    monkeypatch: pytest.MonkeyPatch,
    s3: FakeS3Client,
    vectors: FakeVectorsClient,
) -> None:
    def fake_client(service: str, **kwargs: object):
        del kwargs
        return s3 if service == "s3" else vectors

    monkeypatch.setattr(ensure_data, "client", fake_client)


def test_corpus_ready_requires_every_manifest_digest_and_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        ensure_data.LocalDocument("manifests/a/1.0.json", "digest-a"),
        ensure_data.LocalDocument("manifests/b/1.0.json", "digest-b"),
    ]
    monkeypatch.setattr(ensure_data, "local_documents", lambda: docs)
    s3 = FakeS3Client(
        {
            "manifests/a/1.0.json": {
                "logical_id": "a",
                "version": "1.0",
                "content_sha256": "digest-a",
                "vector_keys": ["a-1", "a-2"],
            },
            "manifests/b/1.0.json": {
                "logical_id": "b",
                "version": "1.0",
                "content_sha256": "digest-b",
                "vector_keys": ["b-1"],
            },
        }
    )
    vectors = FakeVectorsClient({"a-1", "a-2", "b-1"})
    _patch_clients(monkeypatch, s3, vectors)

    assert ensure_data.corpus_ready() is True
    assert {key for call in vectors.calls for key in call} == {"a-1", "a-2", "b-1"}


def test_corpus_ready_rejects_stale_manifest_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ensure_data,
        "local_documents",
        lambda: [ensure_data.LocalDocument("manifests/a/1.0.json", "new-digest")],
    )
    s3 = FakeS3Client(
        {
            "manifests/a/1.0.json": {
                "logical_id": "a",
                "version": "1.0",
                "content_sha256": "old-digest",
                "vector_keys": ["a-1"],
            }
        }
    )
    vectors = FakeVectorsClient({"a-1"})
    _patch_clients(monkeypatch, s3, vectors)

    assert ensure_data.corpus_ready() is False
    assert vectors.calls == []


def test_corpus_ready_rejects_missing_referenced_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ensure_data,
        "local_documents",
        lambda: [ensure_data.LocalDocument("manifests/a/1.0.json", "digest-a")],
    )
    s3 = FakeS3Client(
        {
            "manifests/a/1.0.json": {
                "logical_id": "a",
                "version": "1.0",
                "content_sha256": "digest-a",
                "vector_keys": ["a-1", "a-2"],
            }
        }
    )
    vectors = FakeVectorsClient({"a-1"})
    _patch_clients(monkeypatch, s3, vectors)

    assert ensure_data.corpus_ready() is False
