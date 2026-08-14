from __future__ import annotations

from rag_ops_guard.domain.models import Chunk


def retrieval_text(chunk: Chunk) -> str:
    section = " / ".join(chunk.header_path) or "Document"
    return (
        f"Title: {chunk.title}\n"
        f"System: {chunk.metadata.system}\n"
        f"Document type: {chunk.metadata.document_type.value}\n"
        f"Section: {section}\n\n{chunk.text}"
    )
