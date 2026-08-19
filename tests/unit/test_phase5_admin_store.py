from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from rag_ops_guard.configstore.admin import SecretMutationRequest, SecretMutationStatus
from rag_ops_guard.configstore.admin_store import DynamoDbAdminStore
from rag_ops_guard.configstore.dynamo_store import _decode_item
from rag_ops_guard.configstore.phase3_secret_backend import EnvelopeSecretBackend


class FakeLowLevelDynamo:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _key(raw: dict[str, Any]) -> tuple[str, str]:
        decoded = _decode_item(raw)
        return str(decoded["PK"]), str(decoded["SK"])

    def put_item(self, **kwargs: Any) -> dict[str, object]:
        raw = dict(kwargs["Item"])
        key = self._key(raw)
        if "ConditionExpression" in kwargs and key in self.items:
            raise RuntimeError("conditional write failed")
        self.items[key] = raw
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, object]:
        key = self._key(kwargs["Key"])
        raw = self.items.get(key)
        return {} if raw is None else {"Item": dict(raw)}

    def update_item(self, **kwargs: Any) -> dict[str, object]:
        key = self._key(kwargs["Key"])
        raw = self.items[key]
        decoded = _decode_item(raw)
        values = {
            name: _decode_item({"v": value})["v"]
            for name, value in kwargs["ExpressionAttributeValues"].items()
        }
        condition = kwargs["ConditionExpression"]
        if condition == "#status = :pending AND requested_by <> :approved_by":
            if (
                decoded["status"] != values[":pending"]
                or decoded["requested_by"] == values[":approved_by"]
            ):
                raise RuntimeError("conditional update failed")
            decoded["status"] = values[":processing"]
            decoded["approved_by"] = values[":approved_by"]
        elif condition == "#status = :processing":
            if decoded["status"] != values[":processing"]:
                raise RuntimeError("conditional update failed")
            decoded["status"] = values[":status"]
        else:
            raise AssertionError(condition)
        from rag_ops_guard.configstore.dynamo_store import _item

        self.items[key] = _item(decoded)
        return {"Attributes": dict(self.items[key])}


@dataclass
class FakeEnvelopeService:
    calls: list[tuple[str, str, str, bytes | None]] = field(default_factory=list)

    def set_secret(self, *, scope: str, key_name: str, secret: bytes, kek_ref: str) -> None:
        self.calls.append(("set", scope, key_name, secret))
        assert kek_ref == "kms-key"

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        self.calls.append(("delete", scope, key_name, None))


def _request(
    *,
    request_id: str = "req-1",
    status: SecretMutationStatus = "pending",
) -> SecretMutationRequest:
    return SecretMutationRequest(
        request_id=request_id,
        tenant_id="tenant-a",
        action="set",
        key_name="external_api_key",
        requested_by="operator-a",
        change_reason="rotate credential",
        payload_sha256="a" * 64,
        status=status,
        created_at=123.0,
        kek_ref="kms-key",
    )


def test_dynamo_admin_store_round_trip_claim_finish_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLowLevelDynamo()
    monkeypatch.setattr(
        "rag_ops_guard.configstore.admin_store.boto3.client",
        lambda *args, **kwargs: client,
    )
    store = DynamoDbAdminStore(
        endpoint_url="http://localhost:4566",
        region="us-east-1",
        access_key="test",
        secret_key="test",
        table="rag-ops-config",
        tenant_id="tenant-a",
    )
    request = _request()

    store.create_secret_request(request)
    assert store.get_secret_request("missing") is None
    loaded = store.get_secret_request(request.request_id)
    assert loaded == request
    persisted = _decode_item(client.items[("TENANT#tenant-a", "SECRET_CHANGE#req-1")])
    assert "secret" not in persisted

    claimed = store.claim_secret_request(request.request_id, approved_by="operator-b")
    assert claimed.status == "processing"
    assert claimed.approved_by == "operator-b"
    completed = store.finish_secret_request(request.request_id, status="completed")
    assert completed.status == "completed"

    store.append_secret_audit(completed)
    audit_items = [
        _decode_item(raw)
        for (pk, sk), raw in client.items.items()
        if pk == "TENANT#tenant-a" and sk.startswith("AUDIT#")
    ]
    assert len(audit_items) == 1
    assert audit_items[0]["approving_principal"] == "operator-b"
    assert audit_items[0]["secret_ref"] == "external_api_key"
    assert "payload_sha256" not in audit_items[0]


def test_dynamo_admin_store_rejects_invalid_lifecycle_and_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLowLevelDynamo()
    monkeypatch.setattr(
        "rag_ops_guard.configstore.admin_store.boto3.client",
        lambda *args, **kwargs: client,
    )
    store = DynamoDbAdminStore(
        endpoint_url="",
        region="us-east-1",
        access_key="test",
        secret_key="test",
        table="rag-ops-config",
        tenant_id="tenant-a",
    )

    with pytest.raises(ValueError, match="request_id"):
        store._request_sk("")
    with pytest.raises(ValueError, match="tenant"):
        store.create_secret_request(
            SecretMutationRequest(
                request_id="bad",
                tenant_id="tenant-b",
                action="delete",
                key_name="x",
                requested_by="a",
                change_reason="x",
                payload_sha256=None,
                status="pending",
                created_at=1.0,
            )
        )
    invalid_status = cast(SecretMutationStatus, "pending")
    with pytest.raises(ValueError, match="completed or failed"):
        store.finish_secret_request("missing", status=invalid_status)
    with pytest.raises(ValueError, match="completed"):
        store.append_secret_audit(_request())


def test_phase3_backend_satisfies_generic_secret_port() -> None:
    service = FakeEnvelopeService()
    backend = EnvelopeSecretBackend(service=service, kek_ref="kms-key")

    backend.set_secret(scope="TENANT#tenant-a", key_name="x", secret=b"value")
    backend.delete_secret(scope="TENANT#tenant-a", key_name="x")

    assert service.calls == [
        ("set", "TENANT#tenant-a", "x", b"value"),
        ("delete", "TENANT#tenant-a", "x", None),
    ]
    with pytest.raises(ValueError, match="kek_ref"):
        EnvelopeSecretBackend(service=service, kek_ref="")
