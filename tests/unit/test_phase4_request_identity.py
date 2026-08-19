"""SPEC-P4-MULTITENANCY 4.1 red contracts for authenticated tenant identity."""

from dataclasses import FrozenInstanceError

import pytest

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher
from rag_ops_guard.tenancy.auth import (
    ApiKeyAuthenticationError,
    TenantAuthenticator,
    TenantCredential,
)
from rag_ops_guard.tenancy.context import RequestContext


class FakeCredentialStore:
    def __init__(self, credential: TenantCredential | None) -> None:
        self.credential = credential
        self.requested_key_id: str | None = None

    def get(self, key_id: str) -> TenantCredential | None:
        self.requested_key_id = key_id
        return self.credential


def test_request_context_is_immutable() -> None:
    context = RequestContext(principal="api-key:key-a", tenant_id="tenant-a")
    with pytest.raises(FrozenInstanceError):
        context.tenant_id = "tenant-b"  # type: ignore[misc]


def test_authenticator_derives_tenant_from_verified_credential() -> None:
    hasher = TenantTokenHasher()
    credential = TenantCredential(
        key_id="key-a",
        tenant_id="tenant-a",
        token_hash=hasher.hash_token("correct-secret"),
        enabled=True,
    )
    store = FakeCredentialStore(credential)
    authenticator = TenantAuthenticator(store=store, hasher=hasher)

    context = authenticator.authenticate("key-a.correct-secret")

    assert store.requested_key_id == "key-a"
    assert context == RequestContext(principal="api-key:key-a", tenant_id="tenant-a")


def test_authenticator_rejects_unknown_disabled_or_invalid_credentials() -> None:
    hasher = TenantTokenHasher()
    enabled = TenantCredential(
        key_id="key-a",
        tenant_id="tenant-a",
        token_hash=hasher.hash_token("correct-secret"),
        enabled=True,
    )
    disabled = TenantCredential(
        key_id="key-a",
        tenant_id="tenant-a",
        token_hash=enabled.token_hash,
        enabled=False,
    )

    for store, api_key in (
        (FakeCredentialStore(None), "key-a.correct-secret"),
        (FakeCredentialStore(disabled), "key-a.correct-secret"),
        (FakeCredentialStore(enabled), "key-a.wrong-secret"),
        (FakeCredentialStore(enabled), "malformed"),
    ):
        with pytest.raises(ApiKeyAuthenticationError):
            TenantAuthenticator(store=store, hasher=hasher).authenticate(api_key)
