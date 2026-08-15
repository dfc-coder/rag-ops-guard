from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_ops_guard.domain.models import Citation, QueryResponse, QueryStatus


def _citation() -> Citation:
    return Citation(
        logical_id="calypso-retry",
        title="Payment Retry Policy",
        version="2.0",
        chunk_id="calypso-retry-v2-0",
        s3_key="chunks/calypso-retry/2.0/chunk-000.json",
    )


def test_grounded_answer_requires_citations() -> None:
    with pytest.raises(ValidationError, match="grounded knowledge responses require citations"):
        QueryResponse(
            request_id="r1",
            status=QueryStatus.ANSWERED,
            route="knowledge",
            answer="Three retries.",
            citations=[],
        )


def test_ungrounded_answer_cannot_carry_citations() -> None:
    with pytest.raises(
        ValidationError,
        match="answered_ungrounded responses cannot carry citations",
    ):
        QueryResponse(
            request_id="r2",
            status=QueryStatus.ANSWERED_UNGROUNDED,
            route="chat",
            answer="Here is the Perl code.",
            citations=[_citation()],
        )


def test_direct_answer_without_citations_is_valid() -> None:
    response = QueryResponse(
        request_id="r3",
        status=QueryStatus.ANSWERED_UNGROUNDED,
        route="chat",
        answer="def fib(n): ...",
    )

    assert response.citations == []


def test_grounded_answer_with_citations_is_valid() -> None:
    response = QueryResponse(
        request_id="r4",
        status=QueryStatus.ANSWERED,
        route="knowledge",
        answer="Calypso allows three automated retries.",
        citations=[_citation()],
    )

    assert len(response.citations) == 1


def test_non_answered_states_cannot_carry_citations() -> None:
    with pytest.raises(ValidationError, match="non-answered responses cannot carry citations"):
        QueryResponse(
            request_id="r5",
            status=QueryStatus.INSUFFICIENT_EVIDENCE,
            route="knowledge",
            answer="Not enough evidence.",
            citations=[_citation()],
        )
