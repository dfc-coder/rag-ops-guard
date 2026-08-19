from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

DEK_SORT_PREFIX = "DEK#"
SECRET_SORT_PREFIX = "SECRET#"
SCOPE_PARTITION_PREFIX = "SECRET_SCOPE#"
NONCE_SIZE_BYTES = 12


@dataclass(frozen=True)
class DekRecord:
    scope: str
    version: int
    kek_ref: str
    wrapped_dek: bytes


@dataclass(frozen=True)
class EncryptedSecretRecord:
    scope: str
    key_name: str
    ciphertext: bytes
    nonce: bytes
    dek_version: int


class SecretRecordStore(Protocol):
    def create_dek(self, record: DekRecord) -> None: ...

    def replace_dek(self, record: DekRecord) -> None: ...

    def get_dek(self, *, scope: str, version: int) -> DekRecord | None: ...

    def list_deks(self, *, scope: str) -> list[DekRecord]: ...

    def get_latest_dek(self, *, scope: str) -> DekRecord | None: ...

    def put_secret(self, record: EncryptedSecretRecord) -> None: ...

    def get_secret(self, *, scope: str, key_name: str) -> EncryptedSecretRecord | None: ...

    def list_secrets(self, *, scope: str) -> list[EncryptedSecretRecord]: ...

    def delete_secret(self, *, scope: str, key_name: str) -> None: ...


class DynamoDbSecretStore:
    """Persist wrapped DEKs and encrypted secret envelopes in a PK/SK DynamoDB table."""

    def __init__(self, *, table: Any) -> None:
        self._table = table

    @staticmethod
    def _pk(scope: str) -> str:
        if not scope:
            raise ValueError("scope must not be empty")
        return f"{SCOPE_PARTITION_PREFIX}{scope}"

    @staticmethod
    def _dek_sk(version: int) -> str:
        if version < 1:
            raise ValueError("DEK version must be >= 1")
        return f"{DEK_SORT_PREFIX}{version}"

    @staticmethod
    def _secret_sk(key_name: str) -> str:
        if not key_name:
            raise ValueError("key_name must not be empty")
        return f"{SECRET_SORT_PREFIX}{key_name}"

    @staticmethod
    def _bytes(value: object, *, field: str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        binary_value = getattr(value, "value", None)
        if isinstance(binary_value, (bytes, bytearray)):
            return bytes(binary_value)
        raise ValueError(f"persisted {field} is not binary")

    @classmethod
    def _validate_dek(cls, record: DekRecord) -> None:
        cls._pk(record.scope)
        cls._dek_sk(record.version)
        if not record.kek_ref:
            raise ValueError("kek_ref must not be empty")
        if not record.wrapped_dek:
            raise ValueError("wrapped_dek must not be empty")

    @classmethod
    def _validate_secret(cls, record: EncryptedSecretRecord) -> None:
        cls._pk(record.scope)
        cls._secret_sk(record.key_name)
        cls._dek_sk(record.dek_version)
        if not record.ciphertext:
            raise ValueError("ciphertext must not be empty")
        if len(record.nonce) != NONCE_SIZE_BYTES:
            raise ValueError("secret nonce must be exactly 12 bytes")

    @classmethod
    def _decode_dek(cls, item: dict[str, Any]) -> DekRecord:
        return DekRecord(
            scope=str(item["scope"]),
            version=int(item["dek_version"]),
            kek_ref=str(item["kek_ref"]),
            wrapped_dek=cls._bytes(item["wrapped_dek"], field="wrapped_dek"),
        )

    @classmethod
    def _decode_secret(cls, item: dict[str, Any]) -> EncryptedSecretRecord:
        return EncryptedSecretRecord(
            scope=str(item["scope"]),
            key_name=str(item["key_name"]),
            ciphertext=cls._bytes(item["ciphertext"], field="ciphertext"),
            nonce=cls._bytes(item["nonce"], field="nonce"),
            dek_version=int(item["dek_version"]),
        )

    def create_dek(self, record: DekRecord) -> None:
        self._validate_dek(record)
        self._table.put_item(
            Item={
                "PK": self._pk(record.scope),
                "SK": self._dek_sk(record.version),
                "record_type": "dek",
                "scope": record.scope,
                "dek_version": record.version,
                "kek_ref": record.kek_ref,
                "wrapped_dek": record.wrapped_dek,
            },
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
        )

    def replace_dek(self, record: DekRecord) -> None:
        self._validate_dek(record)
        self._table.put_item(
            Item={
                "PK": self._pk(record.scope),
                "SK": self._dek_sk(record.version),
                "record_type": "dek",
                "scope": record.scope,
                "dek_version": record.version,
                "kek_ref": record.kek_ref,
                "wrapped_dek": record.wrapped_dek,
            }
        )

    def get_dek(self, *, scope: str, version: int) -> DekRecord | None:
        response = self._table.get_item(
            Key={"PK": self._pk(scope), "SK": self._dek_sk(version)},
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not isinstance(item, dict):
            return None
        return self._decode_dek(item)

    def list_deks(self, *, scope: str) -> list[DekRecord]:
        response = self._table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={
                ":pk": self._pk(scope),
                ":prefix": DEK_SORT_PREFIX,
            },
            ConsistentRead=True,
        )
        items = response.get("Items", [])
        records = [self._decode_dek(item) for item in items if isinstance(item, dict)]
        return sorted(records, key=lambda record: record.version)

    def get_latest_dek(self, *, scope: str) -> DekRecord | None:
        records = self.list_deks(scope=scope)
        return records[-1] if records else None

    def put_secret(self, record: EncryptedSecretRecord) -> None:
        self._validate_secret(record)
        self._table.put_item(
            Item={
                "PK": self._pk(record.scope),
                "SK": self._secret_sk(record.key_name),
                "record_type": "secret",
                "scope": record.scope,
                "key_name": record.key_name,
                "ciphertext": record.ciphertext,
                "nonce": record.nonce,
                "dek_version": record.dek_version,
            }
        )

    def get_secret(self, *, scope: str, key_name: str) -> EncryptedSecretRecord | None:
        response = self._table.get_item(
            Key={"PK": self._pk(scope), "SK": self._secret_sk(key_name)},
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not isinstance(item, dict):
            return None
        return self._decode_secret(item)

    def list_secrets(self, *, scope: str) -> list[EncryptedSecretRecord]:
        response = self._table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={
                ":pk": self._pk(scope),
                ":prefix": SECRET_SORT_PREFIX,
            },
            ConsistentRead=True,
        )
        items = response.get("Items", [])
        records = [self._decode_secret(item) for item in items if isinstance(item, dict)]
        return sorted(records, key=lambda record: record.key_name)

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        self._table.delete_item(Key={"PK": self._pk(scope), "SK": self._secret_sk(key_name)})
