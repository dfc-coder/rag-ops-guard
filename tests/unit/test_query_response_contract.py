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


def _citation() -> Citation:
    return Citation(
        logical_id="calypso-retry",
        title="Payment Retry Policy",
        version="2.0",
        chunk_id="calypso-retry-v2-0",
        s3_key="chunks/calypso-retry/2.0/chunk-000.json",
    )


def test_answer_requires_at_least_one_segment() -> None:
    with pytest.raises(ValidationError, match="at least one segment"):
        QueryResponse(request_id="r1", outcome=ResponseOutcome.ANSWER)


def test_non_answer_outcome_cannot_carry_segments() -> None:
    with pytest.raises(ValidationError, match="cannot carry answer segments"):
        QueryResponse(
            request_id="r2",
            outcome=ResponseOutcome.ERROR,
            segments=[ResponseSegment(text="Impossible")],
        )


def test_safety_response_derives_status_without_answer_segments() -> None:
    response = QueryResponse(
        request_id="r3",
        outcome=ResponseOutcome.SAFETY_BLOCKED,
        message="Blocked by deterministic safety.",
        route="safety",
    )

    assert response.status is QueryStatus.SAFETY_BLOCKED
    assert response.answer == "Blocked by deterministic safety."
    assert response.citations == []


def test_grounded_answer_derives_status_answer_and_citations() -> None:
    response = QueryResponse(
        request_id="r4",
        outcome=ResponseOutcome.ANSWER,
        route="knowledge",
        segments=[
            ResponseSegment(
                text="Calypso allows three automated retries.",
                citations=[_citation()],
            )
        ],
    )

    assert response.status is QueryStatus.ANSWERED_GROUNDED
    assert response.answer == "Calypso allows three automated retries."
    assert response.citations == [_citation()]


def test_route_does_not_control_grounding_contract() -> None:
    response = QueryResponse(
        request_id="r5",
        outcome=ResponseOutcome.ANSWER,
        route="chat",
        segments=[ResponseSegment(text="General answer")],
    )

    assert response.status is QueryStatus.ANSWERED_UNGROUNDED
    assert response.citations == []
