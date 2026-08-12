from __future__ import annotations

import os
from pathlib import Path

import httpx


def api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def main() -> None:
    api = api_url()
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        key = f"raw/{path.relative_to('knowledge-base').as_posix()}"
        response = httpx.post(
            f"{api}/v1/ingest",
            json={"s3_key": key},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        print(f"{key}: {payload['status']} ({payload['chunks']} chunks)")


if __name__ == "__main__":
    main()
