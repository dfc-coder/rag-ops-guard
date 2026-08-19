from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.config import Config

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.token_hashing import TenantTokenHasher
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.tenancy import KeyLayout


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required for the Phase 4 physical isolation smoke")
    return value


def _api_url() -> str:
    value = os.environ.get("RAG_API_URL", "").strip()
    if value:
        return value.rstrip("/")
    path = Path(".local/api-url")
    if not path.is_file():
        raise SystemExit(".local/api-url missing; run make up first")
    return path.read_text(encoding="utf-8").strip().rstrip("/")


def _aws(service: str, **kwargs: object) -> Any:
    return boto3.client(
        service,
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        **kwargs,
    )


def _parts(api_key: str) -> tuple[str, str]:
    key_id, separator, secret = api_key.partition(".")
    if not separator or not key_id or not secret:
        raise SystemExit("Phase 4 API keys must use key_id.secret format")
    return key_id, secret


def _put_credential(tenant_id: str, api_key: str) -> None:
    key_id, secret = _parts(api_key)
    _aws("dynamodb").put_item(
        TableName=os.environ.get("TENANT_TABLE", "rag-ops-tenants"),
        Item={
            "key_id": {"S": key_id},
            "tenant_id": {"S": tenant_id},
            "token_hash": {"S": TenantTokenHasher().hash_token(secret)},
            "enabled": {"BOOL": True},
        },
    )


def _put_doc(layout: KeyLayout, marker: str) -> str:
    key = layout.raw_key(f"phase4/{marker}.md")
    body = f"# Phase 4 {marker}\n\nThis document belongs only to {layout.tenant_id}."
    _aws("s3", config=Config(s3={"addressing_style": "path"})).put_object(
        Bucket=os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local"),
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/markdown",
    )
    return key


def _post_ingest(api: str, api_key: str, s3_key: str) -> httpx.Response:
    return httpx.post(
        f"{api}/v1/ingest",
        json={"s3_key": s3_key},
        headers={"x-api-key": api_key},
        timeout=180,
    )


def _post_probe(api: str, api_key: str) -> dict[str, Any]:
    response = httpx.post(
        f"{api}/v1/query",
        json={"__rag_ops_probe__": "connectivity"},
        headers={"x-api-key": api_key},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("connectivity probe returned non-object payload")
    return payload


def _config_head(tenant_id: str) -> int:
    settings = Settings()
    store = TenantDynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
        tenant_id=tenant_id,
    )
    head = store.get_head()
    if head is None:
        raise RuntimeError(f"tenant {tenant_id} has no config HEAD")
    return head.revision_no


def _list_chunks(layout: KeyLayout) -> set[str]:
    response = _aws("s3", config=Config(s3={"addressing_style": "path"})).list_objects_v2(
        Bucket=os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local"),
        Prefix=layout.chunks_prefix,
    )
    return {str(item["Key"]) for item in response.get("Contents", [])}


def main() -> None:
    from scripts.config_publish import publish_config_revision

    api = _api_url()
    tenant_a = os.environ.get("RAG_OPS_TENANT_ID", "tenant-a").strip() or "tenant-a"
    tenant_b = os.environ.get("RAG_OPS_TENANT_ID_B", "tenant-b").strip() or "tenant-b"
    if tenant_a == tenant_b:
        raise SystemExit("RAG_OPS_TENANT_ID and RAG_OPS_TENANT_ID_B must differ")
    key_a = _required("RAG_OPS_API_KEY")
    key_b = _required("RAG_OPS_API_KEY_B")
    layout_a = KeyLayout(tenant_a)
    layout_b = KeyLayout(tenant_b)

    _put_credential(tenant_a, key_a)
    _put_credential(tenant_b, key_b)
    publish_config_revision(
        reason="Phase 4 physical tenant isolation baseline",
        actor=os.environ.get("USER", "local"),
        tenant_id=tenant_b,
    )

    key_doc_a = _put_doc(layout_a, "tenant-a")
    key_doc_b = _put_doc(layout_b, "tenant-b")

    own_a = _post_ingest(api, key_a, key_doc_a)
    own_b = _post_ingest(api, key_b, key_doc_b)
    if own_a.status_code != 200 or own_b.status_code != 200:
        raise RuntimeError(
            f"own-tenant ingestion failed: A={own_a.status_code} {own_a.text[:500]} "
            f"B={own_b.status_code} {own_b.text[:500]}"
        )

    cross_a = _post_ingest(api, key_a, key_doc_b)
    cross_b = _post_ingest(api, key_b, key_doc_a)
    if cross_a.status_code != 400 or cross_b.status_code != 400:
        raise RuntimeError(
            f"cross-tenant ingestion was not rejected: A->B={cross_a.status_code}, "
            f"B->A={cross_b.status_code}"
        )

    probe_a = _post_probe(api, key_a)
    probe_b = _post_probe(api, key_b)
    if probe_a.get("tenant_id") != tenant_a or probe_b.get("tenant_id") != tenant_b:
        raise RuntimeError("connectivity probe tenant identity mismatch")

    expected_index_a = layout_a.vector_index(os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-openvino-v1"))
    expected_index_b = layout_b.vector_index(os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-openvino-v1"))
    actual_index_a = probe_a.get("checks", {}).get("s3vectors", {}).get("index")
    actual_index_b = probe_b.get("checks", {}).get("s3vectors", {}).get("index")
    if actual_index_a != expected_index_a or actual_index_b != expected_index_b:
        raise RuntimeError(
            f"tenant vector indexes mismatch: A={actual_index_a!r}, B={actual_index_b!r}"
        )

    chunks_a = _list_chunks(layout_a)
    chunks_b = _list_chunks(layout_b)
    if not chunks_a or not chunks_b or not chunks_a.isdisjoint(chunks_b):
        raise RuntimeError("tenant chunk prefixes are empty or overlap")

    head_a = _config_head(tenant_a)
    head_b = _config_head(tenant_b)
    if head_a < 1 or head_b < 1:
        raise RuntimeError("tenant config HEAD verification failed")

    print("PHASE 4 TENANT ISOLATION READY")
    print(
        json.dumps(
            {
                "tenant_a": tenant_a,
                "tenant_b": tenant_b,
                "config_head_a": head_a,
                "config_head_b": head_b,
                "vector_index_a": expected_index_a,
                "vector_index_b": expected_index_b,
                "chunks_a": len(chunks_a),
                "chunks_b": len(chunks_b),
                "cross_ingest_a_to_b": cross_a.status_code,
                "cross_ingest_b_to_a": cross_b.status_code,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
