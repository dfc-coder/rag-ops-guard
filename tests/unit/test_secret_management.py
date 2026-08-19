from __future__ import annotations

from typing import Any

import pytest

from rag_ops_guard.configstore.key_provider import LocalKeyProvider
from rag_ops_guard.configstore.secret_service import EnvelopeSecretService
from rag_ops_guard.configstore.secret_store import (
    DekRecord,
    DynamoDbSecretStore,
    EncryptedSecretRecord,
)


class FakeDynamoTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}

    def put_item(self, **kwargs: Any) -> dict[str, object]:
        item = dict(kwargs["Item"])
        key = (str(item["PK"]), str(item["SK"]))
        if "ConditionExpression" in kwargs and key in self.items:
            raise RuntimeError("conditional write failed")
        self.items[key] = item
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, object]:
        key_data = kwargs["Key"]
        key = (str(key_data["PK"]), str(key_data["SK"]))
        item = self.items.get(key)
        return {} if item is None else {"Item": dict(item)}

    def query(self, **kwargs: Any) -> dict[str, object]:
        values = kwargs["ExpressionAttributeValues"]
        pk = str(values[":pk"])
        prefix = str(values[":prefix"])
        items = [
            dict(item)
            for (item_pk, item_sk), item in self.items.items()
            if item_pk == pk and item_sk.startswith(prefix)
        ]
        return {"Items": items}

    def delete_item(self, **kwargs: Any) -> dict[str, object]:
        key_data = kwargs["Key"]
        key = (str(key_data["PK"]), str(key_data["SK"]))
        self.items.pop(key, None)
        return {}


def _store() -> tuple[FakeDynamoTable, DynamoDbSecretStore]:
    table = FakeDynamoTable()
    return table, DynamoDbSecretStore(table=table)


def _service() -> tuple[FakeDynamoTable, DynamoDbSecretStore, EnvelopeSecretService]:
    table, store = _store()
    provider = LocalKeyProvider(
        app_env="local",
        keys={
            "kek-v1": b"a" * 32,
            "kek-v2": b"b" * 32,
        },
    )
    return table, store, EnvelopeSecretService(store=store, key_provider=provider)


def test_scoped_dek_persistence_keeps_only_wrapped_material() -> None:
    table, store = _store()
    record = DekRecord(
        scope="tenant-a",
        version=1,
        kek_ref="kek-v1",
        wrapped_dek=b"wrapped-dek",
    )

    store.create_dek(record)

    assert store.get_dek(scope="tenant-a", version=1) == record
    assert store.get_dek(scope="tenant-b", version=1) is None
    persisted = table.items[("SECRET_SCOPE#tenant-a", "DEK#1")]
    assert persisted["wrapped_dek"] == b"wrapped-dek"
    assert "plaintext" not in persisted
    assert "plaintext_dek" not in persisted


def test_latest_dek_uses_numeric_version_order() -> None:
    _, store = _store()
    for version in (1, 10, 2):
        store.create_dek(
            DekRecord(
                scope="tenant-a",
                version=version,
                kek_ref="kek-v1",
                wrapped_dek=f"wrapped-{version}".encode(),
            )
        )

    latest = store.get_latest_dek(scope="tenant-a")

    assert latest is not None
    assert latest.version == 10


def test_secret_round_trip_uses_scoped_dek() -> None:
    _, store, service = _service()

    service.set_secret(
        scope="tenant-a",
        key_name="openai_api_key",
        secret=b"secret-value",
        kek_ref="kek-v1",
    )

    assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"secret-value"
    assert service.get_secret(scope="tenant-b", key_name="openai_api_key") is None
    dek = store.get_latest_dek(scope="tenant-a")
    assert dek is not None
    assert dek.version == 1


def test_secret_delete_removes_only_the_encrypted_value() -> None:
    _, store, service = _service()
    service.set_secret(
        scope="tenant-a",
        key_name="openai_api_key",
        secret=b"secret-value",
        kek_ref="kek-v1",
    )
    assert store.get_latest_dek(scope="tenant-a") is not None

    service.delete_secret(scope="tenant-a", key_name="openai_api_key")

    assert service.get_secret(scope="tenant-a", key_name="openai_api_key") is None
    assert store.get_latest_dek(scope="tenant-a") is not None


def test_kek_rotation_rewraps_deks_without_touching_secret_ciphertext() -> None:
    _, store, service = _service()
    service.set_secret(
        scope="tenant-a",
        key_name="openai_api_key",
        secret=b"secret-value",
        kek_ref="kek-v1",
    )
    before_secret = store.get_secret(scope="tenant-a", key_name="openai_api_key")
    before_dek = store.get_dek(scope="tenant-a", version=1)
    assert before_secret is not None
    assert before_dek is not None

    rotated = service.rotate_kek(scope="tenant-a", new_kek_ref="kek-v2")

    after_secret = store.get_secret(scope="tenant-a", key_name="openai_api_key")
    after_dek = store.get_dek(scope="tenant-a", version=1)
    assert rotated == 1
    assert after_secret == before_secret
    assert after_dek is not None
    assert after_dek.kek_ref == "kek-v2"
    assert after_dek.wrapped_dek != before_dek.wrapped_dek
    assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"secret-value"


def test_dek_rotation_reencrypts_secrets_and_retains_old_dek() -> None:
    _, store, service = _service()
    service.set_secret(
        scope="tenant-a",
        key_name="openai_api_key",
        secret=b"secret-value",
        kek_ref="kek-v1",
    )
    before = store.get_secret(scope="tenant-a", key_name="openai_api_key")
    assert before is not None

    new_version = service.rotate_dek(scope="tenant-a", kek_ref="kek-v1")

    after = store.get_secret(scope="tenant-a", key_name="openai_api_key")
    assert new_version == 2
    assert store.get_dek(scope="tenant-a", version=1) is not None
    assert store.get_dek(scope="tenant-a", version=2) is not None
    assert after is not None
    assert after.dek_version == 2
    assert after.ciphertext != before.ciphertext
    assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"secret-value"


def test_secret_store_rejects_invalid_record_shapes() -> None:
    _, store = _store()

    with pytest.raises(ValueError, match="version"):
        store.create_dek(
            DekRecord(scope="tenant-a", version=0, kek_ref="kek-v1", wrapped_dek=b"wrapped")
        )

    with pytest.raises(ValueError, match="nonce"):
        store.put_secret(
            EncryptedSecretRecord(
                scope="tenant-a",
                key_name="key",
                ciphertext=b"ciphertext",
                nonce=b"short",
                dek_version=1,
            )
        )
