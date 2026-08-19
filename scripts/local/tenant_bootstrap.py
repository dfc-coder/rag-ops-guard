from __future__ import annotations

import os

import boto3

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher


def _api_key_parts(value: str) -> tuple[str, str]:
    key_id, separator, secret = value.partition(".")
    if not separator or not key_id or not secret:
        raise SystemExit("RAG_OPS_API_KEY must use key_id.secret format")
    return key_id, secret


def main() -> None:
    api_key = os.environ.get("RAG_OPS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RAG_OPS_API_KEY is required for the Phase 4 local runtime")
    tenant_id = os.environ.get("RAG_OPS_TENANT_ID", "default").strip() or "default"
    key_id, secret = _api_key_parts(api_key)

    client = boto3.client(
        "dynamodb",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )
    client.put_item(
        TableName=os.environ.get("TENANT_TABLE", "rag-ops-tenants"),
        Item={
            "key_id": {"S": key_id},
            "tenant_id": {"S": tenant_id},
            "token_hash": {"S": TenantTokenHasher().hash_token(secret)},
            "enabled": {"BOOL": True},
        },
    )
    print(f"tenant credential ready: tenant={tenant_id} key_id={key_id}")


if __name__ == "__main__":
    main()
