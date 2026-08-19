from __future__ import annotations

import time
import uuid
from typing import cast

import boto3

from rag_ops_guard.configstore.admin import (
    SecretMutationAction,
    SecretMutationRequest,
    SecretMutationStatus,
)
from rag_ops_guard.configstore.dynamo_store import _av, _decode_item, _item
from rag_ops_guard.tenancy.config_scope import tenant_config_scope

_REQUEST_PREFIX = "SECRET_CHANGE#"
_AUDIT_PREFIX = "AUDIT#"


class DynamoDbAdminStore:
    """Tenant-scoped approval state; secret plaintext is never accepted by this adapter."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key: str,
        secret_key: str,
        table: str,
        tenant_id: str,
    ) -> None:
        self.table = table
        self.tenant_id = tenant_id
        self.scope_key = tenant_config_scope(tenant_id)
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @staticmethod
    def _request_sk(request_id: str) -> str:
        if not request_id:
            raise ValueError("request_id must not be empty")
        return f"{_REQUEST_PREFIX}{request_id}"

    @staticmethod
    def _decode_request(item: dict[str, object]) -> SecretMutationRequest:
        action = str(item["action"])
        if action not in {"set", "rotate", "delete"}:
            raise ValueError("persisted secret mutation action is invalid")
        status = str(item["status"])
        if status not in {"pending", "processing", "completed", "failed"}:
            raise ValueError("persisted secret mutation status is invalid")
        return SecretMutationRequest(
            request_id=str(item["request_id"]),
            tenant_id=str(item["tenant_id"]),
            action=cast(SecretMutationAction, action),
            key_name=str(item["key_name"]),
            requested_by=str(item["requested_by"]),
            change_reason=str(item["change_reason"]),
            payload_sha256=str(item["payload_sha256"])
            if item.get("payload_sha256") is not None
            else None,
            status=cast(SecretMutationStatus, status),
            created_at=float(str(item["created_at"])),
            approved_by=str(item["approved_by"])
            if item.get("approved_by") is not None
            else None,
            kek_ref=str(item["kek_ref"]) if item.get("kek_ref") is not None else None,
        )

    def create_secret_request(self, request: SecretMutationRequest) -> None:
        if request.tenant_id != self.tenant_id:
            raise ValueError("secret request tenant does not match store tenant")
        item: dict[str, object] = {
            "PK": self.scope_key,
            "SK": self._request_sk(request.request_id),
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "action": request.action,
            "key_name": request.key_name,
            "requested_by": request.requested_by,
            "change_reason": request.change_reason,
            "status": request.status,
            "created_at": str(request.created_at),
        }
        if request.payload_sha256 is not None:
            item["payload_sha256"] = request.payload_sha256
        if request.kek_ref is not None:
            item["kek_ref"] = request.kek_ref
        self._client.put_item(
            TableName=self.table,
            Item=_item(item),
            ConditionExpression="attribute_not_exists(SK)",
        )

    def get_secret_request(self, request_id: str) -> SecretMutationRequest | None:
        response = self._client.get_item(
            TableName=self.table,
            Key=_item({"PK": self.scope_key, "SK": self._request_sk(request_id)}),
            ConsistentRead=True,
        )
        raw = response.get("Item")
        if not raw:
            return None
        return self._decode_request(_decode_item(raw))

    def claim_secret_request(
        self,
        request_id: str,
        *,
        approved_by: str,
    ) -> SecretMutationRequest:
        response = self._client.update_item(
            TableName=self.table,
            Key=_item({"PK": self.scope_key, "SK": self._request_sk(request_id)}),
            UpdateExpression="SET #status = :processing, approved_by = :approved_by",
            ConditionExpression="#status = :pending AND requested_by <> :approved_by",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":pending": _av("pending"),
                ":processing": _av("processing"),
                ":approved_by": _av(approved_by),
            },
            ReturnValues="ALL_NEW",
        )
        raw = response.get("Attributes")
        if not raw:
            raise RuntimeError("secret request claim returned no state")
        return self._decode_request(_decode_item(raw))

    def finish_secret_request(
        self,
        request_id: str,
        *,
        status: SecretMutationStatus,
    ) -> SecretMutationRequest:
        if status not in {"completed", "failed"}:
            raise ValueError("finished secret request status must be completed or failed")
        response = self._client.update_item(
            TableName=self.table,
            Key=_item({"PK": self.scope_key, "SK": self._request_sk(request_id)}),
            UpdateExpression="SET #status = :status",
            ConditionExpression="#status = :processing",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":processing": _av("processing"),
                ":status": _av(status),
            },
            ReturnValues="ALL_NEW",
        )
        raw = response.get("Attributes")
        if not raw:
            raise RuntimeError("secret request completion returned no state")
        return self._decode_request(_decode_item(raw))

    def append_secret_audit(self, request: SecretMutationRequest) -> None:
        if request.status != "completed" or request.approved_by is None:
            raise ValueError("only completed dual-approved secret requests may be audited")
        created = time.time_ns()
        self._client.put_item(
            TableName=self.table,
            Item=_item(
                {
                    "PK": self.scope_key,
                    "SK": f"{_AUDIT_PREFIX}{created}#{uuid.uuid4().hex[:12]}",
                    "tenant_id": self.tenant_id,
                    "actor": request.requested_by,
                    "approving_principal": request.approved_by,
                    "action": f"secret_{request.action}",
                    "change_reason": request.change_reason,
                    "secret_ref": request.key_name,
                    "correlation_id": request.request_id,
                    "published_at": str(created),
                }
            ),
            ConditionExpression="attribute_not_exists(SK)",
        )
