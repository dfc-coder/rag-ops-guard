from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from rag_ops_guard.configstore.hashing import content_hash
from rag_ops_guard.configstore.registry import ConfigScope, RegistryEntry

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()


def _av(value: object) -> Any:
    return _SERIALIZER.serialize(value)


def _item(values: Mapping[str, object]) -> dict[str, Any]:
    return {key: _av(value) for key, value in values.items()}


def _decode_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _DESERIALIZER.deserialize(value) for key, value in item.items()}


def _published_at_seconds(value: object) -> float | None:
    if value is None:
        return None
    raw = int(value) if isinstance(value, Decimal) else int(str(value))
    return raw / 1_000_000_000


@dataclass(frozen=True)
class ConfigHead:
    revision_no: int
    content_hash: str
    published_at: float | None = None


@dataclass(frozen=True)
class PublishedRevision:
    revision_no: int
    content_hash: str


class DynamoDbConfigStore:
    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        access_key: str,
        secret_key: str,
        table: str,
    ) -> None:
        self.table = table
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def ensure_registry(self, entries: dict[str, RegistryEntry]) -> None:
        for name in sorted(entries):
            entry = entries[name]
            item: dict[str, object] = {
                "PK": ConfigScope.REGISTRY.value,
                "SK": f"KEY#{name}",
                "value_type": entry.value_type,
                "json_schema": json.dumps(
                    entry.json_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                "sensitivity": entry.sensitivity,
                "scope": entry.scope,
                "fail_mode": entry.fail_mode,
                "reload_policy": entry.reload_policy,
                "code_default": json.dumps(
                    entry.code_default,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            }
            try:
                self._client.put_item(
                    TableName=self.table,
                    Item=_item(item),
                    ConditionExpression="attribute_not_exists(SK)",
                )
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                    raise

    def get_head(self) -> ConfigHead | None:
        response = self._client.get_item(
            TableName=self.table,
            Key=_item({"PK": ConfigScope.GLOBAL.value, "SK": "HEAD"}),
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
            Key=_item({"PK": ConfigScope.GLOBAL.value, "SK": revision_key}),
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
            _item(
                {
                    "PK": ConfigScope.GLOBAL.value,
                    "SK": f"VAL#{name}#{revision_no:020d}",
                }
            )
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
        if not change_reason.strip():
            raise ValueError("change_reason is required")
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
                                "PK": ConfigScope.GLOBAL.value,
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
        transaction.extend(
            [
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(
                            {
                                "PK": ConfigScope.GLOBAL.value,
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
                        "Item": _item(
                            {
                                "PK": ConfigScope.GLOBAL.value,
                                "SK": f"AUDIT#{published_at}#{uuid.uuid4().hex[:12]}",
                                "actor": actor,
                                "action": "publish",
                                "hash_before": previous.content_hash if previous else "",
                                "hash_after": digest,
                            }
                        ),
                        "ConditionExpression": "attribute_not_exists(SK)",
                    }
                },
                {
                    "Put": {
                        "TableName": self.table,
                        "Item": _item(
                            {
                                "PK": ConfigScope.GLOBAL.value,
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
