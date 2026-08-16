from rag_ops_guard.domain.errors import CitationValidationError
from rag_ops_guard.domain.models import (
    Citation,
    Evidence,
    GeneratedSegment,
    ResponseSegment,
)


def validate_citations(citation_ids: list[str], evidence: list[Evidence]) -> list[Citation]:
    available = {item.chunk.id: item for item in evidence}
    unknown = [citation_id for citation_id in citation_ids if citation_id not in available]
    if unknown:
        raise CitationValidationError(f"unknown citation ids: {unknown}")

    citations: list[Citation] = []
    for citation_id in dict.fromkeys(citation_ids):
        item = available[citation_id]
        chunk = item.chunk
        citations.append(
            Citation(
                logical_id=chunk.logical_id,
                title=chunk.title,
                version=chunk.version,
                chunk_id=chunk.id,
                s3_key=(
                    f"chunks/{chunk.logical_id}/{chunk.version}/chunk-{chunk.chunk_index:03d}.json"
                ),
            )
        )
    return citations


def validate_generated_segments(
    segments: list[GeneratedSegment],
    admitted_evidence: list[Evidence],
) -> list[ResponseSegment]:
    """Materialize public segments only after every model citation ID is admitted."""
    return [
        ResponseSegment(
            text=segment.text,
            citations=validate_citations(segment.citation_ids, admitted_evidence),
        )
        for segment in segments
    ]
