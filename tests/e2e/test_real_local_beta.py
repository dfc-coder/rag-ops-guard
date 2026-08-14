from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _api_url() -> str:
    return os.environ.get("RAG_API_URL") or Path(".local/api-url").read_text().strip()


@pytest.mark.skipif(os.environ.get("RUN_REAL_E2E") != "1", reason="real local E2E is release-only")
def test_real_local_query_is_grounded() -> None:
    response = httpx.post(
        f"{_api_url()}/v1/query",
        json={
            "question": "How many times can a Calypso timeout be retried?",
            "context": {"system": "payments", "environment": "production"},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"]
    assert "three" in payload["answer"].lower() or "3" in payload["answer"]


@pytest.mark.skipif(os.environ.get("RUN_REAL_E2E") != "1", reason="real local E2E is release-only")
def test_broad_named_entity_question_is_answered_in_spanish() -> None:
    response = httpx.post(
        f"{_api_url()}/v1/query",
        json={
            "question": "¿Qué puedes contarme de Calypso?",
            "context": {},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    answer = str(payload.get("answer") or "")
    citation_titles = " ".join(str(item.get("title", "")) for item in payload.get("citations", []))

    assert payload["status"] == "answered"
    assert payload["citations"]
    assert "calypso" in answer.lower()
    assert "calypso" in citation_titles.lower() or "integration landscape" in citation_titles.lower()


@pytest.mark.skipif(os.environ.get("RUN_REAL_E2E") != "1", reason="real local E2E is release-only")
def test_duplicate_payment_incident_id_is_answered() -> None:
    response = httpx.post(
        f"{_api_url()}/v1/query",
        json={
            "question": (
                "Which incident ID was the one where a bulk replay of the payment DLQ "
                "caused duplicate payments?"
            ),
            "context": {"environment": "production"},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    answer = str(payload.get("answer") or "").lower()
    citation_titles = " ".join(str(item.get("title", "")) for item in payload.get("citations", []))

    assert payload["status"] == "answered"
    assert payload["citations"]
    assert "inc-002" in answer or "inc-002" in citation_titles.lower()
