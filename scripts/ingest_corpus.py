from __future__ import annotations

import os
from pathlib import Path

import httpx

from rag_ops_guard.tenancy import KeyLayout


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
    api_key = os.environ.get("RAG_OPS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RAG_OPS_API_KEY is required for Phase 4 ingestion")
    layout = KeyLayout(os.environ.get("RAG_OPS_TENANT_ID", "default"))
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        key = layout.raw_key(path.relative_to("knowledge-base").as_posix())
        response = httpx.post(
            f"{api}/v1/ingest",
            json={"s3_key": key},
            headers={"x-api-key": api_key},
            timeout=180,
        )
        if response.is_error:
            raise RuntimeError(
                f"ingestion failed for {key}: HTTP {response.status_code}: {response.text[:2000]}"
            )
        payload = response.json()
        print(f"{key}: {payload['status']} ({payload['chunks']} chunks)")


if __name__ == "__main__":
    main()
