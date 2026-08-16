from typing import Any

import pytest

from rag_ops_guard.adapters.aws import s3_vectors
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import FakeObjectStore


class RecordingVectorsClient:
    def __init__(self) -> None:
        self.put_payload: dict[str, Any] | None = None

    def put_vectors(self, **kwargs: Any) -> None:
        self.put_payload = kwargs


def test_generic_metadata_can_be_stored_in_real_vector_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC-2.4: missing governance fields cannot break vector persistence."""
    client = RecordingVectorsClient()
    monkeypatch.setattr(s3_vectors.boto3, "client", lambda *args, **kwargs: client)
    generic = metadata(
        doc_id="generic",
        logical_id="generic",
        status=None,
        effective_date=None,
        system=None,
        environment=None,
        document_type=None,
        authority=None,
    )
    item = evidence(meta=generic)
    store = S3VectorsStore(
        "vectors",
        "index",
        FakeObjectStore(),
        "http://floci",
        "us-east-1",
        "test",
        "test",
    )

    store.put([item.chunk], [[1.0, 0.0]])

    assert client.put_payload is not None
    vector_meta = client.put_payload["vectors"][0]["metadata"]
    assert vector_meta["logical_id"] == "generic"
    assert "status" not in vector_meta
    assert "document_type" not in vector_meta
