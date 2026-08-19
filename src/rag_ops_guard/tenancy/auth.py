from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher
from rag_ops_guard.tenancy.context import RequestContext

_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ApiKeyAuthenticationError(ValueError):
    """Raised when a presented tenant API key cannot be authenticated."""


@dataclass(frozen=True, slots=True)
class TenantCredential:
    key_id: str
    tenant_id: str
    token_hash: str
    enabled: bool = True


class TenantCredentialStore(Protocol):
    def get(self, key_id: str) -> TenantCredential | None: ...


class TenantAuthenticator:
    def __init__(self, *, store: TenantCredentialStore, hasher: TenantTokenHasher | None = None) -> None:
        self._store = store
        self._hasher = hasher or TenantTokenHasher()

    def authenticate(self, api_key: str) -> RequestContext:
        key_id, secret = self._split(api_key)
        credential = self._store.get(key_id)
        if credential is None or not credential.enabled:
            raise ApiKeyAuthenticationError("invalid tenant API key")
        if credential.key_id != key_id:
            raise ApiKeyAuthenticationError("invalid tenant API key")
        if not self._hasher.verify_token(encoded_hash=credential.token_hash, token=secret):
            raise ApiKeyAuthenticationError("invalid tenant API key")
        return RequestContext(principal=f"api-key:{key_id}", tenant_id=credential.tenant_id)

    @staticmethod
    def _split(api_key: str) -> tuple[str, str]:
        key_id, separator, secret = api_key.partition(".")
        if not separator or not secret or not _KEY_ID.fullmatch(key_id):
            raise ApiKeyAuthenticationError("invalid tenant API key")
        return key_id, secret
