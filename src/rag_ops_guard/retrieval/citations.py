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
    admitted: list[Evidence] | list[Citation],
) -> list[ResponseSegment]:
    """Materialize public segments only after every model citation ID is admitted."""
    if not admitted:
        available: dict[str, Citation] = {}
    elif isinstance(admitted[0], Evidence):
        evidence = [item for item in admitted if isinstance(item, Evidence)]
        available = {
            citation.chunk_id: citation
            for citation in validate_citations(
                [item.chunk.id for item in evidence],
                evidence,
            )
        }
    else:
        available = {
            citation.chunk_id: citation for citation in admitted if isinstance(citation, Citation)
        }

    requested = [citation_id for segment in segments for citation_id in segment.citation_ids]
    unknown = [citation_id for citation_id in requested if citation_id not in available]
    if unknown:
        raise CitationValidationError(f"unknown citation ids: {list(dict.fromkeys(unknown))}")

    return [
        ResponseSegment(
            text=segment.text,
            citations=[available[citation_id] for citation_id in segment.citation_ids],
        )
        for segment in segments
    ]
