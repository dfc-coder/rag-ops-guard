from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(os.environ.get("RUN_REAL_E2E") != "1", reason="real local E2E is release-only")
def test_real_local_query_is_grounded() -> None:
    api = os.environ.get("RAG_API_URL") or Path(".local/api-url").read_text().strip()
    response = httpx.post(
        f"{api}/v1/query",
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
