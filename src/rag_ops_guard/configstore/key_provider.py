from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEK_SIZE_BYTES = 32
LOCAL_WRAP_NONCE_SIZE_BYTES = 12


@dataclass(frozen=True)
class GeneratedDataKey:
    plaintext: bytes
    wrapped: bytes


class KeyProvider(Protocol):
    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey: ...

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes: ...


class KeyRewrapper(Protocol):
    def rewrap(
        self,
        *,
        source_kek_ref: str,
        target_kek_ref: str,
        wrapped_key: bytes,
    ) -> bytes: ...


class KmsKeyProvider:
    """Key provider backed by a KMS-compatible client."""

    def __init__(self, *, client: Any) -> None:
        self._client = client

    @staticmethod
    def _require_dek_size(value: object) -> bytes:
        if not isinstance(value, bytes) or len(value) != DEK_SIZE_BYTES:
            size = len(value) if isinstance(value, bytes) else 0
            raise ValueError(f"KMS data key must be exactly 32 bytes; got {size}")
        return value

    @staticmethod
    def _require_wrapped(value: object) -> bytes:
        if not isinstance(value, bytes) or not value:
            raise ValueError("KMS returned an empty wrapped data key")
        return value

    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey:
        response = self._client.generate_data_key(KeyId=kek_ref, KeySpec="AES_256")
        plaintext = self._require_dek_size(response.get("Plaintext"))
        wrapped = self._require_wrapped(response.get("CiphertextBlob"))
        return GeneratedDataKey(plaintext=plaintext, wrapped=wrapped)

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes:
        response = self._client.decrypt(KeyId=kek_ref, CiphertextBlob=wrapped_key)
        return self._require_dek_size(response.get("Plaintext"))

    def rewrap(
        self,
        *,
        source_kek_ref: str,
        target_kek_ref: str,
        wrapped_key: bytes,
    ) -> bytes:
        plaintext = self.unwrap(kek_ref=source_kek_ref, wrapped_key=wrapped_key)
        response = self._client.encrypt(KeyId=target_kek_ref, Plaintext=plaintext)
        return self._require_wrapped(response.get("CiphertextBlob"))


class LocalKeyProvider:
    """Local-only provider using injected AES-256 KEKs for DEK wrapping."""

    def __init__(self, *, app_env: str, keys: Mapping[str, bytes] | None = None) -> None:
        if app_env != "local":
            raise RuntimeError("LocalKeyProvider requires app_env=local")
        self._keys = dict(keys or {})
        for kek_ref, key in self._keys.items():
            if len(key) != DEK_SIZE_BYTES:
                raise ValueError(f"local KEK {kek_ref!r} must be exactly 32 bytes")

    def _key(self, kek_ref: str) -> bytes:
        key = self._keys.get(kek_ref)
        if key is None:
            raise ValueError(f"no local KEK configured for {kek_ref!r}")
        return key

    def _wrap(self, *, kek_ref: str, plaintext: bytes) -> bytes:
        nonce = secrets.token_bytes(LOCAL_WRAP_NONCE_SIZE_BYTES)
        ciphertext = AESGCM(self._key(kek_ref)).encrypt(
            nonce,
            plaintext,
            kek_ref.encode("utf-8"),
        )
        return nonce + ciphertext

    def generate_data_key(self, *, kek_ref: str) -> GeneratedDataKey:
        plaintext = secrets.token_bytes(DEK_SIZE_BYTES)
        return GeneratedDataKey(
            plaintext=plaintext,
            wrapped=self._wrap(kek_ref=kek_ref, plaintext=plaintext),
        )

    def unwrap(self, *, kek_ref: str, wrapped_key: bytes) -> bytes:
        if len(wrapped_key) <= LOCAL_WRAP_NONCE_SIZE_BYTES:
            raise ValueError("wrapped local data key is invalid")
        nonce = wrapped_key[:LOCAL_WRAP_NONCE_SIZE_BYTES]
        ciphertext = wrapped_key[LOCAL_WRAP_NONCE_SIZE_BYTES:]
        try:
            plaintext = AESGCM(self._key(kek_ref)).decrypt(
                nonce,
                ciphertext,
                kek_ref.encode("utf-8"),
            )
        except InvalidTag:
            raise ValueError("wrapped local data key authentication failed") from None
        if len(plaintext) != DEK_SIZE_BYTES:
            raise ValueError(f"local data key must be exactly 32 bytes; got {len(plaintext)}")
        return plaintext

    def rewrap(
        self,
        *,
        source_kek_ref: str,
        target_kek_ref: str,
        wrapped_key: bytes,
    ) -> bytes:
        plaintext = self.unwrap(kek_ref=source_kek_ref, wrapped_key=wrapped_key)
        return self._wrap(kek_ref=target_kek_ref, plaintext=plaintext)
