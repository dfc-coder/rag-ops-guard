from __future__ import annotations

import json

from rag_ops_guard.domain.models import IngestResponse, Manifest
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.manifest import document_sha256, load_manifest, save_manifest
from rag_ops_guard.ingestion.metadata import parse_document
from rag_ops_guard.ports import EmbeddingProvider, ObjectStore, VectorStore
from rag_ops_guard.retrieval.text import retrieval_text
from rag_ops_guard.tenancy import KeyLayout


class IngestionService:
    def __init__(
        self,
        object_store: ObjectStore,
        vector_store: VectorStore,
        embeddings: EmbeddingProvider,
        chunker: MarkdownChunker,
        key_layout: KeyLayout | None = None,
    ) -> None:
        self._objects = object_store
        self._vectors = vector_store
        self._embeddings = embeddings
        self._chunker = chunker
        self._keys = key_layout or KeyLayout("default")

    def ingest(self, s3_key: str) -> IngestResponse:
        source_key = self._keys.require_ingest_key(s3_key)
        content = self._objects.get_text(source_key)
        metadata, body = parse_document(content, filename=source_key)
        digest = document_sha256(content)
        previous = load_manifest(
            self._objects,
            metadata.logical_id,
            metadata.version,
            self._keys,
        )

        if previous and previous.content_sha256 == digest:
            return IngestResponse(
                status="no_op",
                document_id=metadata.id,
                logical_id=metadata.logical_id,
                version=metadata.version,
                chunks=len(previous.vector_keys),
            )

        chunk_prefix = self._keys.chunk_prefix(metadata.logical_id, metadata.version)
        previous_chunk_keys = set(self._objects.list_keys(chunk_prefix))
        chunks = self._chunker.split(metadata, body)
        texts = [retrieval_text(chunk) for chunk in chunks]
        vectors = self._embeddings.embed_documents(texts)

        if previous:
            self._vectors.delete(previous.vector_keys)

        current_chunk_keys: set[str] = set()
        for chunk in chunks:
            chunk_key = self._keys.chunk_key(
                metadata.logical_id,
                metadata.version,
                chunk.chunk_index,
            )
            current_chunk_keys.add(chunk_key)
            self._objects.put_text(
                chunk_key,
                json.dumps(chunk.model_dump(mode="json"), indent=2),
                "application/json",
            )

        for stale_key in sorted(previous_chunk_keys - current_chunk_keys):
            self._objects.delete(stale_key)

        vector_keys = self._vectors.put(chunks, vectors)

        save_manifest(
            self._objects,
            Manifest(
                logical_id=metadata.logical_id,
                version=metadata.version,
                content_sha256=digest,
                vector_keys=vector_keys,
            ),
            self._keys,
        )
        return IngestResponse(
            status="ingested",
            document_id=metadata.id,
            logical_id=metadata.logical_id,
            version=metadata.version,
            chunks=len(chunks),
        )
