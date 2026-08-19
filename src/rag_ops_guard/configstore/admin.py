from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from rag_ops_guard.configstore.dynamo_store import ConfigHead, PublishedRevision
from rag_ops_guard.configstore.registry import registry_entries, validate_value_against_schema
from rag_ops_guard.tenancy.config_scope import tenant_config_scope
from rag_ops_guard.tenancy.key_layout import KeyLayout

SecretMutationAction = Literal["set", "rotate", "delete"]
SecretMutationStatus = Literal["pending", "processing", "completed", "failed"]


class AdminAuthorizationError(PermissionError):
    """Raised when an administrative principal is not authorized for a tenant scope."""


@dataclass(frozen=True, slots=True)
class AdminPrincipal:
    principal_id: str
    tenant_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must not be empty")
        for tenant_id in self.tenant_ids:
            KeyLayout(tenant_id)

    def require_tenant(self, tenant_id: str) -> str:
        tenant = KeyLayout(tenant_id).tenant_id
        if tenant not in self.tenant_ids:
            raise AdminAuthorizationError(
                f"principal {self.principal_id!r} is not authorized for tenant {tenant!r}"
            )
        return tenant


@dataclass(frozen=True, slots=True)
class AdminAuditEvent:
    tenant_id: str
    actor: str
    action: str
    change_reason: str
    timestamp: float
    approving_principal: str | None = None
    hash_before: str = ""
    hash_after: str = ""
    source_revision: int | None = None
    resulting_revision: int | None = None
    secret_ref: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigAdminStatus:
    tenant_id: str
    revision_no: int | None
    config_hash: str
    revision_age_s: float | None
    recent_audit: tuple[AdminAuditEvent, ...]


class ConfigAdminStore(Protocol):
    tenant_id: str

    def get_head(self) -> ConfigHead | None: ...

    def get_revision_values(self, revision_no: int) -> dict[str, object]: ...

    def publish_admin(
        self,
        values: dict[str, object],
        *,
        actor: str,
        change_reason: str,
        action: str,
        audit_metadata: Mapping[str, object] | None = None,
    ) -> PublishedRevision: ...

    def list_audit_events(self, *, limit: int = 50) -> list[AdminAuditEvent]: ...


class ConfigAdminService:
    """Tenant-authorized publication, monotonic rollback and read-only status."""

    def __init__(
        self,
        *,
        store_factory: Callable[[str], ConfigAdminStore],
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._store_factory = store_factory
        self._wall_clock = wall_clock

    def _store(self, principal: AdminPrincipal, tenant_id: str) -> ConfigAdminStore:
        tenant = principal.require_tenant(tenant_id)
        store = self._store_factory(tenant)
        if store.tenant_id != tenant:
            raise RuntimeError("administrative store tenant does not match authorized tenant")
        return store

    @staticmethod
    def _validate_values(values: Mapping[str, object]) -> dict[str, object]:
        if not values:
            raise ValueError("configuration revision must contain at least one value")
        entries = registry_entries()
        validated: dict[str, object] = {}
        for name, value in values.items():
            entry = entries.get(name)
            if entry is None or entry.sensitivity == "secret":
                raise ValueError(f"{name}: not publishable")
            validate_value_against_schema(entry, value)
            validated[name] = value
        return validated

    def publish(
        self,
        *,
        principal: AdminPrincipal,
        tenant_id: str,
        values: Mapping[str, object],
        change_reason: str,
    ) -> PublishedRevision:
        if not change_reason.strip():
            raise ValueError("change_reason is required")
        store = self._store(principal, tenant_id)
        validated = self._validate_values(values)
        return store.publish_admin(
            validated,
            actor=principal.principal_id,
            change_reason=change_reason,
            action="publish",
        )

    def rollback(
        self,
        *,
        principal: AdminPrincipal,
        tenant_id: str,
        revision_no: int,
        change_reason: str,
    ) -> PublishedRevision:
        if revision_no < 1:
            raise ValueError("revision_no must be >= 1")
        if not change_reason.strip():
            raise ValueError("change_reason is required")
        store = self._store(principal, tenant_id)
        values = store.get_revision_values(revision_no)
        if not values:
            raise ValueError(f"configuration revision {revision_no} does not exist")
        validated = self._validate_values(values)
        return store.publish_admin(
            validated,
            actor=principal.principal_id,
            change_reason=change_reason,
            action="rollback",
            audit_metadata={"rollback_from_revision": revision_no},
        )

    def status(
        self,
        *,
        principal: AdminPrincipal,
        tenant_id: str,
        audit_limit: int = 20,
    ) -> ConfigAdminStatus:
        if audit_limit < 1 or audit_limit > 100:
            raise ValueError("audit_limit must be between 1 and 100")
        store = self._store(principal, tenant_id)
        head = store.get_head()
        age = None
        if head is not None and head.published_at is not None:
            age = max(0.0, self._wall_clock() - head.published_at)
        return ConfigAdminStatus(
            tenant_id=store.tenant_id,
            revision_no=None if head is None else head.revision_no,
            config_hash="" if head is None else head.content_hash,
            revision_age_s=age,
            recent_audit=tuple(store.list_audit_events(limit=audit_limit)),
        )


@dataclass(frozen=True, slots=True)
class SecretMutationRequest:
    request_id: str
    tenant_id: str
    action: SecretMutationAction
    key_name: str
    requested_by: str
    change_reason: str
    payload_sha256: str | None
    status: SecretMutationStatus
    created_at: float
    approved_by: str | None = None
    kek_ref: str | None = None


class SecretApprovalStore(Protocol):
    tenant_id: str

    def create_secret_request(self, request: SecretMutationRequest) -> None: ...

    def get_secret_request(self, request_id: str) -> SecretMutationRequest | None: ...

    def claim_secret_request(
        self,
        request_id: str,
        *,
        approved_by: str,
    ) -> SecretMutationRequest: ...

    def finish_secret_request(
        self,
        request_id: str,
        *,
        status: SecretMutationStatus,
    ) -> SecretMutationRequest: ...

    def append_secret_audit(self, request: SecretMutationRequest) -> None: ...


class SecretBackend(Protocol):
    def set_secret(self, *, scope: str, key_name: str, secret: bytes) -> None: ...

    def delete_secret(self, *, scope: str, key_name: str) -> None: ...


class SecretMutationService:
    """Two-person secret-change workflow that never persists plaintext."""

    def __init__(
        self,
        *,
        store_factory: Callable[[str], SecretApprovalStore],
        backend_factory: Callable[[SecretMutationRequest], SecretBackend],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store_factory = store_factory
        self._backend_factory = backend_factory
        self._clock = clock

    def _store(self, principal: AdminPrincipal, tenant_id: str) -> SecretApprovalStore:
        tenant = principal.require_tenant(tenant_id)
        store = self._store_factory(tenant)
        if store.tenant_id != tenant:
            raise RuntimeError("secret approval store tenant does not match authorized tenant")
        return store

    @staticmethod
    def _payload_hash(secret: bytes | None) -> str | None:
        return None if secret is None else hashlib.sha256(secret).hexdigest()

    def request(
        self,
        *,
        principal: AdminPrincipal,
        tenant_id: str,
        action: SecretMutationAction,
        key_name: str,
        change_reason: str,
        secret: bytes | None = None,
        kek_ref: str | None = None,
    ) -> SecretMutationRequest:
        store = self._store(principal, tenant_id)
        if not key_name.strip():
            raise ValueError("key_name is required")
        if not change_reason.strip():
            raise ValueError("change_reason is required")
        if action in {"set", "rotate"}:
            if not secret:
                raise ValueError(f"secret payload is required for {action}")
            if not kek_ref:
                raise ValueError(f"kek_ref is required for {action}")
        elif action == "delete":
            if secret is not None:
                raise ValueError("delete secret requests must not carry a secret payload")
            kek_ref = None
        else:
            raise ValueError(f"unsupported secret mutation action {action!r}")

        request = SecretMutationRequest(
            request_id=uuid.uuid4().hex,
            tenant_id=store.tenant_id,
            action=action,
            key_name=key_name,
            requested_by=principal.principal_id,
            change_reason=change_reason,
            payload_sha256=self._payload_hash(secret),
            status="pending",
            created_at=self._clock(),
            kek_ref=kek_ref,
        )
        store.create_secret_request(request)
        return request

    def approve(
        self,
        *,
        principal: AdminPrincipal,
        tenant_id: str,
        request_id: str,
        secret: bytes | None = None,
    ) -> SecretMutationRequest:
        store = self._store(principal, tenant_id)
        request = store.get_secret_request(request_id)
        if request is None:
            raise ValueError("secret mutation request does not exist")
        if request.tenant_id != store.tenant_id:
            raise AdminAuthorizationError("secret mutation request belongs to another tenant")
        if request.requested_by == principal.principal_id:
            raise AdminAuthorizationError("secret mutation approval requires a distinct principal")
        if request.status != "pending":
            raise ValueError("secret mutation request is not pending")

        if request.action in {"set", "rotate"}:
            if not secret:
                raise ValueError("secret payload is required for approval")
            if self._payload_hash(secret) != request.payload_sha256:
                raise ValueError("approval secret payload does not match the requested payload")
        elif secret is not None:
            raise ValueError("delete secret approval must not carry a secret payload")

        claimed = store.claim_secret_request(request_id, approved_by=principal.principal_id)
        backend = self._backend_factory(claimed)
        scope = tenant_config_scope(claimed.tenant_id)
        try:
            if claimed.action == "delete":
                backend.delete_secret(scope=scope, key_name=claimed.key_name)
            else:
                assert secret is not None
                backend.set_secret(scope=scope, key_name=claimed.key_name, secret=secret)
        except Exception:
            store.finish_secret_request(request_id, status="failed")
            raise

        completed = store.finish_secret_request(request_id, status="completed")
        store.append_secret_audit(completed)
        return completed
