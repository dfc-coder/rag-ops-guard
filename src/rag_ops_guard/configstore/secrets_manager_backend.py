from __future__ import annotations

import base64
from dataclasses import dataclass
from time import monotonic
from typing import Any

import boto3
from botocore.exceptions import ClientError


class RecoverableSecretError(RuntimeError):
    pass


def secret_name(scope: str, key_name: str) -> str:
    clean_scope = scope.strip().replace("#", "/").replace(" ", "-").lower()
    clean_key = key_name.strip().replace(" ", "-").lower()
    if not clean_scope or not clean_key or ".." in clean_scope or ".." in clean_key:
        raise ValueError("secret scope and key name must be non-empty canonical identifiers")
    return f"rag-ops-guard/{clean_scope}/{clean_key}"


class SecretsManagerSecretBackend:
    """Phase-6 authoritative backend for recoverable secrets."""

    def __init__(self, *, client: Any | None = None) -> None:
        self._client = client or boto3.client("secretsmanager")

    def set_secret(self, *, scope: str, key_name: str, secret: bytes) -> None:
        if not secret:
            raise ValueError("recoverable secret must not be empty")
        name = secret_name(scope, key_name)
        try:
            self._client.put_secret_value(SecretId=name, SecretBinary=secret)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"ResourceNotFoundException", "ResourceNotFound"}:
                raise
            self._client.create_secret(
                Name=name,
                SecretBinary=secret,
                Description="RAG Ops Guard recoverable runtime secret",
            )

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        name = secret_name(scope, key_name)
        try:
            self._client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"ResourceNotFoundException", "ResourceNotFound"}:
                raise


@dataclass(slots=True)
class _CachedSecret:
    value: bytes
    expires_at: float


class SecretsManagerResolver:
    """Read recoverable runtime secrets with a bounded in-memory TTL cache."""

    def __init__(self, *, client: Any | None = None, ttl_seconds: float = 60.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("secret cache ttl must be > 0")
        self._client = client or boto3.client("secretsmanager")
        self._ttl = ttl_seconds
        self._cache: dict[str, _CachedSecret] = {}

    def get_bytes(self, secret_ref: str) -> bytes:
        ref = secret_ref.strip()
        if not ref:
            raise ValueError("secret reference must not be empty")
        now = monotonic()
        cached = self._cache.get(ref)
        if cached is not None and cached.expires_at > now:
            return cached.value
        response = self._client.get_secret_value(SecretId=ref)
        if response.get("SecretBinary") is not None:
            raw = response["SecretBinary"]
            value = bytes(raw) if not isinstance(raw, str) else base64.b64decode(raw)
        elif response.get("SecretString") is not None:
            value = str(response["SecretString"]).encode("utf-8")
        else:
            raise RecoverableSecretError(f"Secrets Manager returned no value for {ref!r}")
        if not value:
            raise RecoverableSecretError(f"Secrets Manager returned an empty value for {ref!r}")
        self._cache[ref] = _CachedSecret(value=value, expires_at=now + self._ttl)
        if len(self._cache) > 32:
            oldest = min(self._cache, key=lambda key: self._cache[key].expires_at)
            self._cache.pop(oldest, None)
        return value

    def get_text(self, secret_ref: str) -> str:
        return self.get_bytes(secret_ref).decode("utf-8")
