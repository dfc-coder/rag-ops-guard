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
        logical_id="policy",
        title="Policy",
        version="2.0",
        chunk_id="policy:2.0:000:deadbeef",
        s3_key="chunks/policy/2.0/chunk-000.json",
    )


def test_answer_response_requires_at_least_one_segment() -> None:
    with pytest.raises(ValidationError, match="at least one segment"):
        QueryResponse(request_id="1", outcome=ResponseOutcome.ANSWER)


def test_route_is_telemetry_not_grounding_policy() -> None:
    for route in ("chat", "capabilities", "catalog", "out_of_scope", "uncertain"):
        response = QueryResponse(
            request_id="1",
            outcome=ResponseOutcome.ANSWER,
            route=route,
            segments=[ResponseSegment(text="Deterministic application response.")],
        )
        assert response.status is QueryStatus.ANSWERED_UNGROUNDED
        assert response.citations == []


def test_clarification_requires_question() -> None:
    with pytest.raises(ValidationError, match="clarification_question"):
        QueryResponse(
            request_id="1",
            outcome=ResponseOutcome.CLARIFICATION_REQUIRED,
        )


def test_grounded_segment_derives_grounded_response_contract() -> None:
    response = QueryResponse(
        request_id="1",
        outcome=ResponseOutcome.ANSWER,
        route="knowledge",
        segments=[ResponseSegment(text="Three retries.", citations=[_citation()])],
    )

    assert response.status is QueryStatus.ANSWERED_GROUNDED
    assert response.citations == [_citation()]
