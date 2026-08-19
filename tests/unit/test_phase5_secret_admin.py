from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from rag_ops_guard.configstore.admin import (
    AdminAuthorizationError,
    AdminPrincipal,
    SecretMutationRequest,
    SecretMutationService,
)


@dataclass
class FakeSecretApprovalStore:
    tenant_id: str
    requests: dict[str, SecretMutationRequest] = field(default_factory=dict)
    audit: list[dict[str, object]] = field(default_factory=list)

    def create_secret_request(self, request: SecretMutationRequest) -> None:
        assert request.request_id not in self.requests
        self.requests[request.request_id] = request

    def get_secret_request(self, request_id: str) -> SecretMutationRequest | None:
        return self.requests.get(request_id)

    def claim_secret_request(self, request_id: str, *, approved_by: str) -> SecretMutationRequest:
        request = self.requests[request_id]
        if request.status != "pending":
            raise ValueError("secret mutation request is not pending")
        claimed = replace(request, status="processing", approved_by=approved_by)
        self.requests[request_id] = claimed
        return claimed

    def finish_secret_request(self, request_id: str, *, status: str) -> SecretMutationRequest:
        request = replace(self.requests[request_id], status=status)
        self.requests[request_id] = request
        return request

    def append_secret_audit(self, request: SecretMutationRequest) -> None:
        self.audit.append(
            {
                "tenant_id": request.tenant_id,
                "actor": request.requested_by,
                "approver": request.approved_by,
                "action": f"secret_{request.action}",
                "secret_ref": request.key_name,
                "change_reason": request.change_reason,
            }
        )


@dataclass
class FakeSecretBackend:
    values: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def set_secret(self, *, scope: str, key_name: str, secret: bytes) -> None:
        self.values[(scope, key_name)] = secret

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        self.values.pop((scope, key_name), None)


def _principal(name: str, *tenants: str) -> AdminPrincipal:
    return AdminPrincipal(principal_id=name, tenant_ids=frozenset(tenants))


def test_secret_request_persists_digest_only_and_requires_distinct_approver() -> None:
    store = FakeSecretApprovalStore("tenant-a")
    backend = FakeSecretBackend()
    service = SecretMutationService(
        store_factory=lambda _tenant: store,
        backend_factory=lambda _request: backend,
    )
    secret = b"super-secret-do-not-persist"

    request = service.request(
        principal=_principal("operator-a", "tenant-a"),
        tenant_id="tenant-a",
        action="set",
        key_name="langsmith_api_key",
        change_reason="rotate external token",
        secret=secret,
        kek_ref="kms-key-1",
    )

    persisted = store.requests[request.request_id]
    assert persisted.payload_sha256
    assert secret.decode() not in repr(persisted)
    assert secret not in repr(store.requests).encode()

    with pytest.raises(AdminAuthorizationError, match="distinct"):
        service.approve(
            principal=_principal("operator-a", "tenant-a"),
            tenant_id="tenant-a",
            request_id=request.request_id,
            secret=secret,
        )
    assert backend.values == {}

    completed = service.approve(
        principal=_principal("operator-b", "tenant-a"),
        tenant_id="tenant-a",
        request_id=request.request_id,
        secret=secret,
    )
    assert completed.status == "completed"
    assert backend.values[("TENANT#tenant-a", "langsmith_api_key")] == secret
    assert secret.decode() not in repr(store.audit)
    assert store.audit[-1]["approver"] == "operator-b"


def test_secret_approval_rejects_wrong_payload_and_cross_tenant_before_backend() -> None:
    stores = {"tenant-a": FakeSecretApprovalStore("tenant-a")}
    backend = FakeSecretBackend()
    service = SecretMutationService(
        store_factory=lambda tenant: stores.setdefault(tenant, FakeSecretApprovalStore(tenant)),
        backend_factory=lambda _request: backend,
    )
    request = service.request(
        principal=_principal("operator-a", "tenant-a"),
        tenant_id="tenant-a",
        action="rotate",
        key_name="external_api_key",
        change_reason="scheduled rotation",
        secret=b"new-value",
        kek_ref="kms-key-1",
    )

    with pytest.raises(ValueError, match="does not match"):
        service.approve(
            principal=_principal("operator-b", "tenant-a"),
            tenant_id="tenant-a",
            request_id=request.request_id,
            secret=b"different-value",
        )
    assert backend.values == {}

    with pytest.raises(AdminAuthorizationError):
        service.approve(
            principal=_principal("operator-b", "tenant-b"),
            tenant_id="tenant-a",
            request_id=request.request_id,
            secret=b"new-value",
        )
    assert backend.values == {}


def test_secret_delete_is_also_dual_controlled() -> None:
    store = FakeSecretApprovalStore("tenant-a")
    backend = FakeSecretBackend({("TENANT#tenant-a", "legacy_key"): b"old"})
    service = SecretMutationService(
        store_factory=lambda _tenant: store,
        backend_factory=lambda _request: backend,
    )

    request = service.request(
        principal=_principal("operator-a", "tenant-a"),
        tenant_id="tenant-a",
        action="delete",
        key_name="legacy_key",
        change_reason="credential retired",
    )
    completed = service.approve(
        principal=_principal("operator-b", "tenant-a"),
        tenant_id="tenant-a",
        request_id=request.request_id,
    )

    assert completed.status == "completed"
    assert ("TENANT#tenant-a", "legacy_key") not in backend.values
    assert store.audit[-1]["action"] == "secret_delete"
