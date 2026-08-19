from __future__ import annotations

import math
from typing import Any

import boto3

from rag_ops_guard.domain.models import Chunk, Evidence
from rag_ops_guard.ports import ObjectStore
from rag_ops_guard.tenancy import KeyLayout


class S3VectorsStore:
    def __init__(
        self,
        vector_bucket: str,
        index_name: str,
        object_store: ObjectStore,
        endpoint_url: str,
        region_name: str,
        access_key: str,
        secret_key: str,
        key_layout: KeyLayout | None = None,
    ) -> None:
        self._bucket = vector_bucket
        self._keys = key_layout or KeyLayout("default")
        self._index = index_name
        self._objects = object_store
        self._client: Any = boto3.client(
            "s3vectors",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def put(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        vectors: list[dict[str, object]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not embedding or not all(math.isfinite(value) for value in embedding):
                raise ValueError("embedding must contain finite values")
            chunk_s3_key = self._keys.chunk_key(
                chunk.logical_id,
                chunk.version,
                chunk.chunk_index,
            )
            vectors.append(
                {
                    "key": chunk.id,
                    "data": {"float32": [float(value) for value in embedding]},
                    "metadata": _vector_metadata(chunk, chunk_s3_key),
                }
            )
        if vectors:
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=vectors,
            )
        return [chunk.id for chunk in chunks]

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[Evidence]:
        params: dict[str, object] = {
            "vectorBucketName": self._bucket,
            "indexName": self._index,
            "topK": top_k,
            "queryVector": {"float32": [float(value) for value in embedding]},
            "returnMetadata": True,
            "returnDistance": True,
        }
        if filters:
            params["filter"] = filters
        response = self._client.query_vectors(**params)
        evidence: list[Evidence] = []
        for result in response.get("vectors", []):
            metadata = result.get("metadata") or {}
            chunk_s3_key = metadata.get("chunk_s3_key")
            if not isinstance(chunk_s3_key, str) or not self._keys.owns(chunk_s3_key):
                continue
            chunk = Chunk.model_validate_json(self._objects.get_text(chunk_s3_key))
            evidence.append(Evidence(chunk=chunk, distance=result.get("distance")))
        return evidence

    def delete(self, keys: list[str]) -> None:
        if not keys:
            return
        self._client.delete_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            keys=keys,
        )


def _vector_metadata(chunk: Chunk, chunk_s3_key: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "logical_id": chunk.logical_id,
        "version": chunk.version,
        "chunk_index": chunk.chunk_index,
        "chunk_s3_key": chunk_s3_key,
    }
    document = chunk.metadata
    optional = {
        "status": document.status.value if document.status else None,
        "system": document.system,
        "environment": document.environment,
        "document_type": document.document_type.value if document.document_type else None,
    }
    metadata.update({key: value for key, value in optional.items() if value is not None})
    return metadata
