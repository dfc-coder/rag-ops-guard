from __future__ import annotations

import os
from pathlib import Path

from rag_ops_guard.app import ingestion_service
from rag_ops_guard.tenancy import KeyLayout, RequestContext


def main() -> None:
    tenant_id = os.environ.get("RAG_OPS_TENANT_ID", "default")
    context = RequestContext(principal="local:demo-prepare", tenant_id=tenant_id)
    layout = KeyLayout(tenant_id)
    service = ingestion_service(context)
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        key = layout.raw_key(path.relative_to("knowledge-base").as_posix())
        response = service.ingest(key)
        print(f"{key}: {response.status} ({response.chunks} chunks)")


if __name__ == "__main__":
    main()
