from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedDataKey:
    plaintext: bytes
    wrapped: bytes


class KeyProvider(Protocol):
    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey: ...

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes: ...


class KmsKeyProvider:
    """KeyProvider backed by a KMS-compatible client.

    The client is injected deliberately so the same port can be exercised against
    Floci in local integration tests and AWS KMS in production without changing
    envelope-encryption code.
    """

    def __init__(self, *, client: Any) -> None:
        self._client = client

    @staticmethod
    def _require_dek_size(value: object) -> bytes:
        if not isinstance(value, bytes) or len(value) != 32:
            size = len(value) if isinstance(value, bytes) else 0
            raise ValueError(f"KMS data key must be exactly 32 bytes; got {size}")
        return value

    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey:
        response = self._client.generate_data_key(KeyId=kek_ref, KeySpec="AES_256")
        plaintext = self._require_dek_size(response.get("Plaintext"))
        wrapped = response.get("CiphertextBlob")
        if not isinstance(wrapped, bytes) or not wrapped:
            raise ValueError("KMS GenerateDataKey returned an empty wrapped data key")
        return GeneratedDataKey(plaintext=plaintext, wrapped=wrapped)

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes:
        response = self._client.decrypt(KeyId=kek_ref, CiphertextBlob=wrapped_key)
        return self._require_dek_size(response.get("Plaintext"))


class LocalKeyProvider:
    """Local-only provider boundary.

    Local wrapping is implemented together with the AES-256-GCM envelope slice so
    no temporary or weaker cryptographic construction is introduced here.
    """

    def __init__(self, *, app_env: str) -> None:
        if app_env != "local":
            raise RuntimeError("LocalKeyProvider requires app_env=local")

    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey:
        raise NotImplementedError("local key wrapping is implemented with the envelope cipher")

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes:
        raise NotImplementedError("local key wrapping is implemented with the envelope cipher")
