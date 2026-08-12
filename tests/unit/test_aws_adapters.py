from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError

from rag_ops_guard.adapters.aws import s3_store, s3_vectors
from rag_ops_guard.adapters.aws.s3_store import S3ObjectStore
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from tests.fixtures.builders import evidence
from tests.fixtures.fakes import FakeObjectStore


class FakeBody:
    def __init__(self, value: bytes) -> None:
        self._value = BytesIO(value)

    def read(self) -> bytes:
        return self._value.read()


class FakeS3Client:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        return {"Body": FakeBody(self.values[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        del Bucket, ContentType
        self.values[Key] = Body

    def head_object(self, *, Bucket: str, Key: str) -> None:
        del Bucket
        if Key not in self.values:
            raise ClientError(
                {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
                "HeadObject",
            )


class FakeVectorsClient:
    def __init__(self) -> None:
        self.put_payload: dict[str, Any] | None = None
        self.deleted: list[str] = []
        self.query_response: dict[str, Any] = {"vectors": []}

    def put_vectors(self, **kwargs: Any) -> None:
        self.put_payload = kwargs

    def query_vectors(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return self.query_response

    def delete_vectors(self, **kwargs: Any) -> None:
        self.deleted.extend(kwargs["keys"])


def test_s3_object_store_roundtrip_and_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeS3Client()
    monkeypatch.setattr(s3_store.boto3, "client", lambda *args, **kwargs: fake)
    store = S3ObjectStore("docs", "http://floci", "us-east-1", "test", "test")

    assert store.exists("missing") is False
    store.put_text("raw/doc.md", "hello", "text/markdown")
    assert store.exists("raw/doc.md") is True
    assert store.get_text("raw/doc.md") == "hello"


def test_s3_vectors_put_query_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVectorsClient()
    monkeypatch.setattr(s3_vectors.boto3, "client", lambda *args, **kwargs: fake)
    objects = FakeObjectStore()
    item = evidence()
    chunk_key = (
        f"chunks/{item.chunk.logical_id}/{item.chunk.version}/"
        f"chunk-{item.chunk.chunk_index:03d}.json"
    )
    objects.put_text(chunk_key, item.chunk.model_dump_json())
    fake.query_response = {
        "vectors": [
            {
                "key": item.chunk.id,
                "distance": 0.05,
                "metadata": {"chunk_s3_key": chunk_key},
            }
        ]
    }
    store = S3VectorsStore(
        "vectors",
        "index",
        objects,
        "http://floci",
        "us-east-1",
        "test",
        "test",
    )

    keys = store.put([item.chunk], [[1.0, 0.0, 0.0, 0.0]])
    assert keys == [item.chunk.id]
    assert fake.put_payload is not None
    assert fake.put_payload["vectors"][0]["metadata"]["logical_id"] == item.chunk.logical_id

    result = store.query([1.0, 0.0, 0.0, 0.0], 1)
    assert result[0].chunk.id == item.chunk.id
    assert result[0].distance == 0.05

    store.delete(keys)
    assert fake.deleted == keys


def test_s3_vectors_reject_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVectorsClient()
    monkeypatch.setattr(s3_vectors.boto3, "client", lambda *args, **kwargs: fake)
    store = S3VectorsStore(
        "vectors",
        "index",
        FakeObjectStore(),
        "http://floci",
        "us-east-1",
        "test",
        "test",
    )
    item = evidence()
    with pytest.raises(ValueError, match="length mismatch"):
        store.put([item.chunk], [])
    with pytest.raises(ValueError, match="finite"):
        store.put([item.chunk], [[float("nan")]])
