from rag_ops_guard.domain.models import QueryResponse, QueryStatus, ResponseOutcome


def test_query_response_carries_config_hash() -> None:
    response = QueryResponse(
        request_id="req-1",
        outcome=ResponseOutcome.INSUFFICIENT_EVIDENCE,
        message="Configuration safety policy is stale.",
        route="uncertain",
        config_hash="a" * 64,
    )

    payload = response.model_dump(mode="json")

    assert response.status == QueryStatus.INSUFFICIENT_EVIDENCE
    assert payload["config_hash"] == "a" * 64
    assert payload["status"] == "insufficient_evidence"
