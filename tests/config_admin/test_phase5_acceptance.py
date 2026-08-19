from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import pytest

from rag_ops_guard.configstore.admin import (
    AdminAuditEvent,
    AdminAuthorizationError,
    AdminPrincipal,
    ConfigAdminService,
    SecretMutationRequest,
    SecretMutationService,
)
from rag_ops_guard.configstore.dynamo_store import ConfigHead, PublishedRevision
from rag_ops_guard.configstore.hashing import content_hash


@dataclass
class MemoryStore:
    tenant_id: str
    revisions: dict[int, dict[str, object]] = field(default_factory=dict)
    audit: list[AdminAuditEvent] = field(default_factory=list)
    secret_requests: dict[str, SecretMutationRequest] = field(default_factory=dict)

    def get_head(self) -> ConfigHead | None:
        if not self.revisions:
            return None
        number = max(self.revisions)
        return ConfigHead(number, content_hash(self.revisions[number]), 100.0)

    def get_revision_values(self, revision_no: int) -> dict[str, object]:
        return dict(self.revisions.get(revision_no, {}))

    def publish_admin(
        self,
        values: dict[str, object],
        *,
        actor: str,
        change_reason: str,
        action: str,
        audit_metadata: Mapping[str, object] | None = None,
    ) -> PublishedRevision:
        before = self.get_head()
        number = 1 if before is None else before.revision_no + 1
        self.revisions[number] = dict(values)
        digest = content_hash(values)
        metadata = dict(audit_metadata or {})
        self.audit.append(
            AdminAuditEvent(
                tenant_id=self.tenant_id,
                actor=actor,
                action=action,
                change_reason=change_reason,
                timestamp=float(number),
                hash_before="" if before is None else before.content_hash,
                hash_after=digest,
                source_revision=metadata.get("rollback_from_revision"),
                resulting_revision=number,
            )
        )
        return PublishedRevision(number, digest)

    def list_audit_events(self, *, limit: int = 50) -> list[AdminAuditEvent]:
        return list(reversed(self.audit[-limit:]))

    def create_secret_request(self, request: SecretMutationRequest) -> None:
        self.secret_requests[request.request_id] = request

    def get_secret_request(self, request_id: str) -> SecretMutationRequest | None:
        return self.secret_requests.get(request_id)

    def claim_secret_request(self, request_id: str, *, approved_by: str) -> SecretMutationRequest:
        current = self.secret_requests[request_id]
        if current.status != "pending":
            raise ValueError("secret mutation request is not pending")
        current = replace(current, status="processing", approved_by=approved_by)
        self.secret_requests[request_id] = current
        return current

    def finish_secret_request(self, request_id: str, *, status: str) -> SecretMutationRequest:
        current = replace(self.secret_requests[request_id], status=status)
        self.secret_requests[request_id] = current
        return current

    def append_secret_audit(self, request: SecretMutationRequest) -> None:
        self.audit.append(
            AdminAuditEvent(
                tenant_id=request.tenant_id,
                actor=request.requested_by,
                approving_principal=request.approved_by,
                action=f"secret_{request.action}",
                change_reason=request.change_reason,
                timestamp=200.0,
                secret_ref=request.key_name,
                correlation_id=request.request_id,
            )
        )


@dataclass
class MemorySecrets:
    values: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def set_secret(self, *, scope: str, key_name: str, secret: bytes) -> None:
        self.values[(scope, key_name)] = secret

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        self.values.pop((scope, key_name), None)


def principal(name: str, tenant: str) -> AdminPrincipal:
    return AdminPrincipal(principal_id=name, tenant_ids=frozenset({tenant}))


def test_bdd_phase5_publish_rollback_dual_control_and_cross_tenant_denial() -> None:
    stores: dict[str, MemoryStore] = {}

    def store_for(tenant: str) -> MemoryStore:
        return stores.setdefault(tenant, MemoryStore(tenant))

    config = ConfigAdminService(store_factory=store_for, wall_clock=lambda: 300.0)
    operator_a = principal("operator-a", "tenant-a")
    first = config.publish(
        principal=operator_a,
        tenant_id="tenant-a",
        values={"retrieval_top_k": 10},
        change_reason="baseline",
    )
    config.publish(
        principal=operator_a,
        tenant_id="tenant-a",
        values={"retrieval_top_k": 20},
        change_reason="temporary change",
    )
    rolled = config.rollback(
        principal=operator_a,
        tenant_id="tenant-a",
        revision_no=first.revision_no,
        change_reason="restore known configuration",
    )
    assert rolled.revision_no == 3
    assert stores["tenant-a"].revisions == {
        1: {"retrieval_top_k": 10},
        2: {"retrieval_top_k": 20},
        3: {"retrieval_top_k": 10},
    }

    secrets = MemorySecrets()
    secret_admin = SecretMutationService(
        store_factory=store_for,
        backend_factory=lambda _request: secrets,
    )
    request = secret_admin.request(
        principal=operator_a,
        tenant_id="tenant-a",
        action="set",
        key_name="external_api_key",
        change_reason="credential rotation",
        secret=b"phase5-secret",
        kek_ref="kms-key",
    )
    with pytest.raises(AdminAuthorizationError):
        secret_admin.approve(
            principal=operator_a,
            tenant_id="tenant-a",
            request_id=request.request_id,
            secret=b"phase5-secret",
        )
    secret_admin.approve(
        principal=principal("operator-b", "tenant-a"),
        tenant_id="tenant-a",
        request_id=request.request_id,
        secret=b"phase5-secret",
    )
    assert secrets.values[("TENANT#tenant-a", "external_api_key")] == b"phase5-secret"
    assert "phase5-secret" not in repr(stores["tenant-a"].audit)

    with pytest.raises(AdminAuthorizationError):
        config.publish(
            principal=operator_a,
            tenant_id="tenant-b",
            values={"retrieval_top_k": 30},
            change_reason="must be denied",
        )
    assert "tenant-b" not in stores
