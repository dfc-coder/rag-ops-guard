import pytest
from pydantic import ValidationError

from rag_ops_guard.domain.models import Citation, QueryResponse, QueryStatus


def test_answered_response_requires_citation_by_default() -> None:
    with pytest.raises(ValidationError):
        QueryResponse(request_id="1", status=QueryStatus.ANSWERED, answer="answer")


def test_chat_answer_does_not_require_citation() -> None:
    response = QueryResponse(
        request_id="1",
        status=QueryStatus.ANSWERED,
        route="chat",
        answer="Soy un asistente de operaciones.",
    )
    assert response.citations == []


def test_clarification_requires_question() -> None:
    with pytest.raises(ValidationError):
        QueryResponse(request_id="1", status=QueryStatus.CLARIFICATION_REQUIRED)


def test_answered_response_accepts_grounded_contract() -> None:
    response = QueryResponse(
        request_id="1",
        status=QueryStatus.ANSWERED,
        route="knowledge",
        answer="Three retries.",
        citations=[
            Citation(
                logical_id="policy",
                title="Policy",
                version="2.0",
                chunk_id="policy:2.0:000:deadbeef",
                s3_key="chunks/policy/2.0/chunk-000.json",
            )
        ],
    )
    assert response.status == QueryStatus.ANSWERED
