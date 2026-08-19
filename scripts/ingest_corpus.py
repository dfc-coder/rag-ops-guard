from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

from rag_ops_guard.local_credentials import require_local_api_key
from rag_ops_guard.tenancy import KeyLayout


def api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the repository corpus for one tenant")
    parser.add_argument("--tenant-id", default="default")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tenant_id = str(args.tenant_id)
    api = api_url()
    api_key = require_local_api_key(tenant_id)
    layout = KeyLayout(tenant_id)
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
