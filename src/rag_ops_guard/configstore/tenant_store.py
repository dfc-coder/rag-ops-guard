from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, cast

import boto3

from rag_ops_guard.configstore.admin import AdminAuditEvent
from rag_ops_guard.configstore.dynamo_store import (
    ConfigHead,
    DynamoDbConfigStore,
    PublishedRevision,
    _av,
    _decode_item,
    _item,
    _published_at_seconds,
)
from rag_ops_guard.configstore.hashing import content_hash
from rag_ops_guard.tenancy.config_scope import tenant_config_scope


class TenantDynamoDbConfigStore(DynamoDbConfigStore):
    """Append-only config store whose data plane is isolated by tenant partition key."""

    def __init__(
        self,
        *,
        table: str,
        tenant_id: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self.table = table
        self.tenant_id = tenant_id
        self.scope_key = tenant_config_scope(tenant_id)
        client_kwargs: dict[str, object] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if region:
            client_kwargs["region_name"] = region
        if access_key is not None:
            client_kwargs["aws_access_key_id"] = access_key
        if secret_key is not None:
            client_kwargs["aws_secret_access_key"] = secret_key
        self._client = boto3.client("dynamodb", **client_kwargs)

    def get_head(self) -> ConfigHead | None:
        response = self._client.get_item(
            TableName=self.table,
            Key=_item({"PK": self.scope_key, "SK": "HEAD"}),
            ConsistentRead=True,
        )
        raw = response.get("Item")
        if not raw:
            return None
        item = _decode_item(raw)
        revision_raw = item["revision_no"]
        if isinstance(revision_raw, Decimal):
            revision_no = int(revision_raw)
        elif isinstance(revision_raw, int):
            revision_no = revision_raw
        else:
            revision_no = int(str(revision_raw))
        return ConfigHead(
            revision_no=revision_no,
            content_hash=str(item["content_hash"]),
            published_at=_published_at_seconds(item.get("published_at")),
        )

    def get_revision_values(self, revision_no: int) -> dict[str, object]:
        revision_key = f"REV#{revision_no:020d}"
        response = self._client.get_item(
            TableName=self.table,
            Key=_item({"PK": self.scope_key, "SK": revision_key}),
            ConsistentRead=True,
        )
        raw_revision = response.get("Item")
        if not raw_revision:
            return {}
        revision = _decode_item(raw_revision)
        raw_key_names = revision.get("key_names", [])
        if not isinstance(raw_key_names, list):
            return {}
        key_names = [str(name) for name in raw_key_names]
        if not key_names:
            return {}
        keys = [
            _item({"PK": self.scope_key, "SK": f"VAL#{name}#{revision_no:020d}"})
            for name in key_names
        ]
        response = self._client.batch_get_item(
            RequestItems={self.table: {"Keys": keys, "ConsistentRead": True}}
        )
        values: dict[str, object] = {}
        raw_items = response.get("Responses", {}).get(self.table, [])
        for raw in cast(list[dict[str, Any]], raw_items):
            item = _decode_item(raw)
            sk = str(item["SK"])
            name = sk[len("VAL#") : -(len(f"#{revision_no:020d}"))]
            values[name] = json.loads(str(item["value_json"]))
        return values

    def publish(
        self,
        values: dict[str, object],
        *,
        actor: str,
        change_reason: str,
    ) -> PublishedRevision:
        return self.publish_admin(
            values,
            actor=actor,
            change_reason=change_reason,
            action="publish",
        )

    def publish_admin(
        self,
        values: dict[str, object],
        *,
        actor: str,
        change_reason: str,
        action: str,
        audit_metadata: Mapping[str, object] | None = None,
    ) -> PublishedRevision:
        if not change_reason.strip():
            raise ValueError("change_reason is required")
        if action not in {"publish", "rollback"}:
            raise ValueError(f"unsupported config admin action {action!r}")
        previous = self.get_head()
        revision_no = 1 if previous is None else previous.revision_no + 1
        digest = content_hash(values)
        suffix = f"{revision_no:020d}"
        transaction: list[dict[str, Any]] = []

        for name in sorted(values):
            transaction.append(
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(
                            {
                                "PK": self.scope_key,
                                "SK": f"VAL#{name}#{suffix}",
                                "value_json": json.dumps(
                                    values[name],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                ),
                            }
                        ),
                        "ConditionExpression": "attribute_not_exists(SK)",
                    }
                }
            )

        published_at = time.time_ns()
        audit: dict[str, object] = {
            "PK": self.scope_key,
            "SK": f"AUDIT#{published_at}#{uuid.uuid4().hex[:12]}",
            "tenant_id": self.tenant_id,
            "actor": actor,
            "action": action,
            "change_reason": change_reason,
            "hash_before": previous.content_hash if previous else "",
            "hash_after": digest,
            "resulting_revision": revision_no,
            "published_at": str(published_at),
        }
        for key, value in dict(audit_metadata or {}).items():
            if key not in audit and value is not None:
                audit[key] = value

        transaction.extend(
            [
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(
                            {
                                "PK": self.scope_key,
                                "SK": f"REV#{suffix}",
                                "content_hash": digest,
                                "key_names": sorted(values),
                                "published_by": actor,
                                "change_reason": change_reason,
                                "published_at": str(published_at),
                            }
                        ),
                        "ConditionExpression": "attribute_not_exists(SK)",
                    }
                },
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(audit),
                        "ConditionExpression": "attribute_not_exists(SK)",
                    }
                },
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(
                            {
                                "PK": self.scope_key,
                                "SK": "HEAD",
                                "revision_no": revision_no,
                                "content_hash": digest,
                                "published_at": str(published_at),
                            }
                        ),
                        "ConditionExpression": (
                            "attribute_not_exists(SK) OR revision_no < :revision_no"
                        ),
                        "ExpressionAttributeValues": {":revision_no": _av(revision_no)},
                    }
                },
            ]
        )
        self._client.transact_write_items(TransactItems=transaction)
        return PublishedRevision(revision_no=revision_no, content_hash=digest)

    def list_audit_events(self, *, limit: int = 50) -> list[AdminAuditEvent]:
        if limit < 1 or limit > 100:
            raise ValueError("audit limit must be between 1 and 100")
        response = self._client.query(
            TableName=self.table,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={
                ":pk": _av(self.scope_key),
                ":prefix": _av("AUDIT#"),
            },
            ScanIndexForward=False,
            Limit=limit,
            ConsistentRead=True,
        )
        events: list[AdminAuditEvent] = []
        for raw in cast(list[dict[str, Any]], response.get("Items", [])):
            item = _decode_item(raw)
            published_at = _published_at_seconds(item.get("published_at")) or 0.0
            source = item.get("rollback_from_revision")
            resulting = item.get("resulting_revision")
            events.append(
                AdminAuditEvent(
                    tenant_id=str(item.get("tenant_id") or self.tenant_id),
                    actor=str(item.get("actor") or "unknown"),
                    action=str(item.get("action") or "unknown"),
                    change_reason=str(item.get("change_reason") or ""),
                    timestamp=published_at,
                    approving_principal=str(item["approving_principal"])
                    if item.get("approving_principal") is not None
                    else None,
                    hash_before=str(item.get("hash_before") or ""),
                    hash_after=str(item.get("hash_after") or ""),
                    source_revision=int(source) if source is not None else None,
                    resulting_revision=int(resulting) if resulting is not None else None,
                    secret_ref=str(item["secret_ref"])
                    if item.get("secret_ref") is not None
                    else None,
                    correlation_id=str(item["correlation_id"])
                    if item.get("correlation_id") is not None
                    else None,
                )
            )
        return events
