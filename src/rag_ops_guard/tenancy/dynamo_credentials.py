from __future__ import annotations

from typing import Any

import boto3
from boto3.dynamodb.types import TypeDeserializer

from rag_ops_guard.tenancy.auth import TenantCredential

_DESERIALIZER = TypeDeserializer()


class DynamoDbTenantCredentialStore:
    """Lookup-only credential store; cleartext API-key secrets are never persisted."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key: str,
        secret_key: str,
        table: str,
    ) -> None:
        self._table = table
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def get(self, key_id: str) -> TenantCredential | None:
        response = self._client.get_item(
            TableName=self._table,
            Key={"key_id": {"S": key_id}},
            ConsistentRead=True,
        )
        raw = response.get("Item")
        if not raw:
            return None
        item: dict[str, Any] = {
            name: _DESERIALIZER.deserialize(value) for name, value in raw.items()
        }
        return TenantCredential(
            key_id=str(item["key_id"]),
            tenant_id=str(item["tenant_id"]),
            token_hash=str(item["token_hash"]),
            enabled=bool(item.get("enabled", True)),
        )
