from __future__ import annotations

import json

from rag_ops_guard.domain.models import IngestResponse, Manifest
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.manifest import document_sha256, load_manifest, save_manifest
from rag_ops_guard.ingestion.metadata import parse_document
from rag_ops_guard.ports import EmbeddingProvider, ObjectStore, VectorStore
from rag_ops_guard.retrieval.text import retrieval_text


class IngestionService:
    def __init__(
        self,
        object_store: ObjectStore,
        vector_store: VectorStore,
        embeddings: EmbeddingProvider,
        chunker: MarkdownChunker,
    ) -> None:
        self._objects = object_store
        self._vectors = vector_store
        self._embeddings = embeddings
        self._chunker = chunker

    def ingest(self, s3_key: str) -> IngestResponse:
        content = self._objects.get_text(s3_key)
        metadata, body = parse_document(content)
        digest = document_sha256(content)
        previous = load_manifest(self._objects, metadata.logical_id, metadata.version)

        if previous and previous.content_sha256 == digest:
            return IngestResponse(
                status="no_op",
                document_id=metadata.id,
                logical_id=metadata.logical_id,
                version=metadata.version,
                chunks=len(previous.vector_keys),
            )

        chunks = self._chunker.split(metadata, body)
        texts = [retrieval_text(chunk) for chunk in chunks]
        vectors = self._embeddings.embed_documents(texts)

        if previous:
            self._vectors.delete(previous.vector_keys)

        for chunk in chunks:
            self._objects.put_text(
                f"chunks/{chunk.logical_id}/{chunk.version}/chunk-{chunk.chunk_index:03d}.json",
                json.dumps(chunk.model_dump(mode="json"), indent=2),
                "application/json",
            )
        vector_keys = self._vectors.put(chunks, vectors)

        save_manifest(
            self._objects,
            Manifest(
                logical_id=metadata.logical_id,
                version=metadata.version,
                content_sha256=digest,
                vector_keys=vector_keys,
            ),
        )
        return IngestResponse(
            status="ingested",
            document_id=metadata.id,
            logical_id=metadata.logical_id,
            version=metadata.version,
            chunks=len(chunks),
        )
