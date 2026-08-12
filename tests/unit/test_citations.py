import pytest

from rag_ops_guard.domain.errors import CitationValidationError
from rag_ops_guard.retrieval.citations import validate_citations
from tests.fixtures.builders import evidence


def test_citation_must_reference_resolved_evidence() -> None:
    item = evidence()
    citations = validate_citations([item.chunk.id], [item])
    assert citations[0].chunk_id == item.chunk.id


def test_unknown_citation_is_rejected() -> None:
    with pytest.raises(CitationValidationError, match="unknown citation"):
        validate_citations(["unknown"], [evidence()])
