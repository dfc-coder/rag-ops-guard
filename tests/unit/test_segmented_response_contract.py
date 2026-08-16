from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_ops_guard.domain.models import (
    Citation,
    QueryResponse,
    QueryStatus,
    ResponseOutcome,
    ResponseSegment,
)
from rag_ops_guard.retrieval.citations import CitationValidationError, validate_generated_segments
from rag_ops_guard.domain.models import GeneratedSegment
from tests.fixtures.builders import evidence, metadata


def _citation(chunk_id: str = "chunk-001") -> Citation:
    return Citation(
        logical_id="doc",
        title="Document",
        version="1.0",
        chunk_id=chunk_id,
        s3_key=f"chunks/doc/1.0/{chunk_id}.json",
    )


def test_segment_grounding_is_derived_from_citations() -> None:
    """SPEC-1.1 / SPEC-1.3: grounded state cannot contradict citation state."""
    plain = ResponseSegment(text="General explanation.")
    grounded = ResponseSegment(text="Document fact.", citations=[_citation()])

    assert plain.grounded is False
    assert grounded.grounded is True

    with pytest.raises(ValidationError, match="grounded"):
        ResponseSegment(text="Impossible", citations=[_citation()], grounded=False)


def test_answer_status_is_derived_not_assigned() -> None:
    """SPEC-1.4"""
    with pytest.raises(ValidationError, match="status"):
        QueryResponse(
            request_id="r",
            outcome=ResponseOutcome.ANSWER,
            segments=[ResponseSegment(text="General explanation.")],
            status=QueryStatus.ANSWERED_UNGROUNDED,
        )


def test_mixed_response_reports_mixed_status_and_derived_citations() -> None:
    """SPEC-1.4 / SPEC-1.5"""
    response = QueryResponse(
        request_id="r",
        outcome=ResponseOutcome.ANSWER,
        segments=[
            ResponseSegment(text="No document supports the SAP timeout."),
            ResponseSegment(text="Calypso allows three retries.", citations=[_citation()]),
        ],
    )

    assert response.status is QueryStatus.ANSWERED_MIXED
    assert response.answer == (
        "No document supports the SAP timeout.\n\nCalypso allows three retries."
    )
    assert response.citations == [_citation()]


def test_grounded_and_ungrounded_statuses_are_derived_from_segments() -> None:
    """SPEC-1.4"""
    grounded = QueryResponse(
        request_id="g",
        outcome=ResponseOutcome.ANSWER,
        segments=[ResponseSegment(text="Fact", citations=[_citation()])],
    )
    ungrounded = QueryResponse(
        request_id="u",
        outcome=ResponseOutcome.ANSWER,
        segments=[ResponseSegment(text="General answer")],
    )

    assert grounded.status is QueryStatus.ANSWERED_GROUNDED
    assert ungrounded.status is QueryStatus.ANSWERED_UNGROUNDED


def test_generated_citation_must_reference_admitted_chunk() -> None:
    """SPEC-1.2: invented chunk IDs fail validation."""
    item = evidence(meta=metadata(doc_id="doc-v1", logical_id="doc"), text="Evidence")

    with pytest.raises(CitationValidationError, match="chunk-999"):
        validate_generated_segments(
            [GeneratedSegment(text="Claim", citation_ids=["chunk-999"])],
            [item],
        )
