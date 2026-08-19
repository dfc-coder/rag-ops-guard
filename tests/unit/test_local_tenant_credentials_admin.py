from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher
from rag_ops_guard.local_credentials import TenantApiCredential
from scripts.local import tenant_credentials as admin


class FakeDynamo:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        return {}


@dataclass
class FakeStore:
    value: str | None = None
    deleted: bool = False

    def get_api_key(self, tenant_id: str) -> str | None:
        assert tenant_id == "tenant-a"
        return self.value

    def require_api_key(self, tenant_id: str) -> str:
        value = self.get_api_key(tenant_id)
        if value is None:
            raise RuntimeError("missing")
        return value

    def set_api_key(self, tenant_id: str, api_key: str) -> None:
        assert tenant_id == "tenant-a"
        self.value = api_key
        self.deleted = False

    def delete_api_key(self, tenant_id: str) -> None:
        assert tenant_id == "tenant-a"
        self.value = None
        self.deleted = True


def test_register_persists_only_argon2id_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamo = FakeDynamo()
    monkeypatch.setattr(admin, "_client", lambda service: dynamo)
    monkeypatch.setattr(admin, "_tenant_table", lambda: "rag-ops-tenants")
    credential = TenantApiCredential(key_id="key-a", secret="super-secret")

    admin._register(tenant_id="tenant-a", credential=credential)

    assert len(dynamo.put_calls) == 1
    call = dynamo.put_calls[0]
    item = call["Item"]
    assert call["TableName"] == "rag-ops-tenants"
    assert item["key_id"] == {"S": "key-a"}
    assert item["tenant_id"] == {"S": "tenant-a"}
    assert item["enabled"] == {"BOOL": True}
    encoded = item["token_hash"]["S"]
    assert encoded.startswith("$argon2id$")
    assert TenantTokenHasher().verify_token(encoded_hash=encoded, token="super-secret")
    assert "super-secret" not in json.dumps(call)


def test_create_keeps_generated_secret_out_of_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeStore()
    registered: list[TenantApiCredential] = []
    monkeypatch.setattr(admin, "_ensure_key_id_available", lambda **kwargs: None)
    monkeypatch.setattr(
        admin,
        "_register",
        lambda *, tenant_id, credential: registered.append(credential),
    )
    monkeypatch.setattr(admin.secrets, "token_urlsafe", lambda size: "generated-secret")

    admin._create(tenant_id="tenant-a", key_id="key-a", store=store)  # type: ignore[arg-type]

    assert store.value == "key-a.generated-secret"
    assert registered == [TenantApiCredential(key_id="key-a", secret="generated-secret")]
    output = capsys.readouterr().out
    assert "generated-secret" not in output
    assert '"server_plaintext_persisted": false' in output


def test_create_removes_local_secret_if_server_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    monkeypatch.setattr(admin, "_ensure_key_id_available", lambda **kwargs: None)
    monkeypatch.setattr(admin.secrets, "token_urlsafe", lambda size: "generated-secret")

    def fail_register(*, tenant_id: str, credential: TenantApiCredential) -> None:
        del tenant_id, credential
        raise RuntimeError("ddb unavailable")

    monkeypatch.setattr(admin, "_register", fail_register)

    with pytest.raises(RuntimeError, match="ddb unavailable"):
        admin._create(tenant_id="tenant-a", key_id="key-a", store=store)  # type: ignore[arg-type]

    assert store.value is None
    assert store.deleted is True
