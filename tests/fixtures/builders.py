from datetime import date

from rag_ops_guard.domain.models import (
    Chunk,
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    Evidence,
)


def metadata(
    *,
    doc_id: str = "payment-retry-policy-v2",
    logical_id: str = "payment-retry-policy",
    version: str = "2.0",
    status: DocumentStatus | None = DocumentStatus.ACTIVE,
    effective_date: date | None = date(2026, 6, 1),
    system: str | None = "payments",
    environment: str | None = "production",
    document_type: DocumentType | None = DocumentType.RUNBOOK,
    authority: int | None = 100,
    supersedes: list[str] | None = None,
) -> DocumentMetadata:
    return DocumentMetadata(
        id=doc_id,
        logical_id=logical_id,
        title="Payment Retry Policy",
        version=version,
        status=status,
        effective_date=effective_date,
        system=system,
        environment=environment,
        document_type=document_type,
        authority=authority,
        supersedes=supersedes or [],
    )


def evidence(
    *,
    meta: DocumentMetadata | None = None,
    index: int = 0,
    distance: float = 0.1,
    text: str = "Retry a Calypso timeout at most 3 times.",
) -> Evidence:
    resolved_meta = meta or metadata()
    chunk = Chunk(
        id=f"{resolved_meta.logical_id}:{resolved_meta.version}:{index:03d}:deadbeef",
        logical_id=resolved_meta.logical_id,
        version=resolved_meta.version,
        title=resolved_meta.title,
        header_path=["Timeout Handling"],
        chunk_index=index,
        text=text,
        metadata=resolved_meta,
        content_sha256="deadbeef" * 8,
    )
    return Evidence(chunk=chunk, distance=distance)
