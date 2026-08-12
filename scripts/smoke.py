from __future__ import annotations

import json
import os
from pathlib import Path

import httpx


def main() -> None:
    api = os.environ.get("RAG_API_URL")
    if not api:
        path = Path(".local/api-url")
        if not path.exists():
            raise SystemExit("API URL not found; run make local-provision")
        api = path.read_text().strip()

    ingest = httpx.post(
        f"{api}/v1/ingest",
        json={"s3_key": "raw/runbooks/payment-retry-v2.md"},
        timeout=180,
    )
    ingest.raise_for_status()
    print("ingest:", json.dumps(ingest.json(), indent=2))

    query = httpx.post(
        f"{api}/v1/query",
        json={
            "question": "How many times can a Calypso timeout be retried?",
            "context": {"system": "payments", "environment": "production"},
        },
        timeout=180,
    )
    query.raise_for_status()
    payload = query.json()
    print("query:", json.dumps(payload, indent=2))
    if payload["status"] != "answered" or not payload["citations"]:
        raise SystemExit("smoke test failed")


if __name__ == "__main__":
    main()
