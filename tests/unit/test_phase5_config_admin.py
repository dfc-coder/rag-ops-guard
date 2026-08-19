from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pytest

from rag_ops_guard.configstore.admin import (
    AdminAuditEvent,
    AdminAuthorizationError,
    AdminPrincipal,
    ConfigAdminService,
)
from rag_ops_guard.configstore.dynamo_store import ConfigHead, PublishedRevision
from rag_ops_guard.configstore.hashing import content_hash


@dataclass
class FakeConfigAdminStore:
    tenant_id: str
    revisions: dict[int, dict[str, object]] = field(default_factory=dict)
    audits: list[AdminAuditEvent] = field(default_factory=list)

    def get_head(self) -> ConfigHead | None:
        if not self.revisions:
            return None
        revision = max(self.revisions)
        values = self.revisions[revision]
        return ConfigHead(revision, content_hash(values), 1_700_000_000.0)

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
        previous = self.get_head()
        revision = 1 if previous is None else previous.revision_no + 1
        self.revisions[revision] = dict(values)
        digest = content_hash(values)
        metadata = dict(audit_metadata or {})
        self.audits.append(
            AdminAuditEvent(
                tenant_id=self.tenant_id,
                actor=actor,
                action=action,
                change_reason=change_reason,
                timestamp=1_700_000_000.0 + revision,
                hash_before="" if previous is None else previous.content_hash,
                hash_after=digest,
                source_revision=int(metadata["rollback_from_revision"])
                if "rollback_from_revision" in metadata
                else None,
                resulting_revision=revision,
            )
        )
        return PublishedRevision(revision, digest)

    def list_audit_events(self, *, limit: int = 50) -> list[AdminAuditEvent]:
        return list(reversed(self.audits[-limit:]))


def _principal(*tenants: str, name: str = "operator-a") -> AdminPrincipal:
    return AdminPrincipal(principal_id=name, tenant_ids=frozenset(tenants))


def test_publish_requires_authorized_tenant_reason_and_non_secret_schema() -> None:
    stores: dict[str, FakeConfigAdminStore] = {}

    def factory(tenant_id: str) -> FakeConfigAdminStore:
        return stores.setdefault(tenant_id, FakeConfigAdminStore(tenant_id))

    service = ConfigAdminService(store_factory=factory)
    values = {
        "retrieval_top_k": 12,
        "retrieval_context_k": 4,
        "retrieval_domain_min_relevance": 0.5,
        "retrieval_min_relevance": 0.5,
    }

    published = service.publish(
        principal=_principal("tenant-a"),
        tenant_id="tenant-a",
        values=values,
        change_reason="raise candidate budget",
    )
    assert published.revision_no == 1
    assert stores["tenant-a"].audits[0].actor == "operator-a"
    assert stores["tenant-a"].audits[0].change_reason == "raise candidate budget"

    with pytest.raises(AdminAuthorizationError):
        service.publish(
            principal=_principal("tenant-a"),
            tenant_id="tenant-b",
            values=values,
            change_reason="cross tenant",
        )
    assert "tenant-b" not in stores

    with pytest.raises(ValueError, match="change_reason"):
        service.publish(
            principal=_principal("tenant-a"),
            tenant_id="tenant-a",
            values=values,
            change_reason=" ",
        )
    with pytest.raises(ValueError, match="not publishable"):
        service.publish(
            principal=_principal("tenant-a"),
            tenant_id="tenant-a",
            values={"langsmith_api_key": "plaintext"},
            change_reason="must fail",
        )


def test_rollback_creates_new_revision_and_never_moves_head_backwards() -> None:
    store = FakeConfigAdminStore("tenant-a")
    service = ConfigAdminService(store_factory=lambda _tenant: store)
    principal = _principal("tenant-a")

    first = service.publish(
        principal=principal,
        tenant_id="tenant-a",
        values={"retrieval_top_k": 10},
        change_reason="baseline",
    )
    second = service.publish(
        principal=principal,
        tenant_id="tenant-a",
        values={"retrieval_top_k": 20},
        change_reason="experiment",
    )
    rolled = service.rollback(
        principal=principal,
        tenant_id="tenant-a",
        revision_no=first.revision_no,
        change_reason="restore baseline",
    )

    assert second.revision_no == 2
    assert rolled.revision_no == 3
    assert store.get_head() is not None and store.get_head().revision_no == 3
    assert store.get_revision_values(1) == {"retrieval_top_k": 10}
    assert store.get_revision_values(2) == {"retrieval_top_k": 20}
    assert store.get_revision_values(3) == {"retrieval_top_k": 10}
    assert store.audits[-1].action == "rollback"
    assert store.audits[-1].source_revision == 1
    assert store.audits[-1].resulting_revision == 3


def test_status_exposes_revision_hash_age_and_recent_audit_without_values() -> None:
    store = FakeConfigAdminStore("tenant-a")
    service = ConfigAdminService(store_factory=lambda _tenant: store, wall_clock=lambda: 1_700_000_100.0)
    principal = _principal("tenant-a")
    service.publish(
        principal=principal,
        tenant_id="tenant-a",
        values={"retrieval_top_k": 10},
        change_reason="baseline",
    )

    status = service.status(principal=principal, tenant_id="tenant-a")

    assert status.tenant_id == "tenant-a"
    assert status.revision_no == 1
    assert len(status.config_hash) == 64
    assert status.revision_age_s == 100.0
    assert status.recent_audit[0].action == "publish"
