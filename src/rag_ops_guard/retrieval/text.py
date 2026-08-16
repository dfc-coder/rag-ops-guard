from __future__ import annotations

from rag_ops_guard.domain.models import Chunk


def retrieval_text(chunk: Chunk) -> str:
    section = " / ".join(chunk.header_path) or "Document"
    lines = [f"Title: {chunk.title}"]
    if chunk.metadata.system:
        lines.append(f"System: {chunk.metadata.system}")
    if chunk.metadata.document_type:
        lines.append(f"Document type: {chunk.metadata.document_type.value}")
    lines.append(f"Section: {section}")
    return "\n".join(lines) + f"\n\n{chunk.text}"
