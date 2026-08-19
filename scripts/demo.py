from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx

from rag_ops_guard.local_credentials import require_local_api_key

ROOT = Path(__file__).resolve().parents[1]
API_FILE = ROOT / ".local" / "api-url"


def _api_url() -> str:
    configured = os.environ.get("RAG_API_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    if API_FILE.is_file():
        return API_FILE.read_text(encoding="utf-8").strip().rstrip("/")
    raise SystemExit("API local no provisionada. Ejecutá `make up` primero.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one RAG Ops Guard query through the canonical Floci/Lambda data plane"
    )
    parser.add_argument("question")
    parser.add_argument("--system")
    parser.add_argument("--environment", choices=["production", "staging"])
    parser.add_argument("--thread-id")
    parser.add_argument("--tenant-id", default="default")
    args = parser.parse_args()

    context = {
        key: value
        for key, value in {
            "system": args.system,
            "environment": args.environment,
        }.items()
        if value is not None
    }
    payload = {
        "question": args.question,
        "context": context,
        "thread_id": args.thread_id or str(uuid4()),
    }

    timeout = float(os.environ.get("DEMO_HTTP_TIMEOUT_SECONDS", "300"))
    response = httpx.post(
        f"{_api_url()}/v1/query",
        json=payload,
        headers={"x-api-key": require_local_api_key(str(args.tenant_id))},
        timeout=timeout,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"error": "non_json_response", "body": response.text}

    print(json.dumps(body, indent=2, ensure_ascii=False))
    response.raise_for_status()


if __name__ == "__main__":
    main()
