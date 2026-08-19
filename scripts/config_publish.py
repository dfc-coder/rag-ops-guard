from __future__ import annotations

import argparse
import os

from botocore.exceptions import BotoCoreError, ClientError

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import PublishedRevision
from rag_ops_guard.configstore.registry import (
    public_settings_values,
    registry_entries,
    validate_value_against_schema,
)
from rag_ops_guard.configstore.runtime import snapshot_for_values, write_local_snapshot
from rag_ops_guard.configstore.tenant_runtime import (
    tenant_snapshot_key,
    upload_tenant_snapshot,
    write_tenant_snapshot,
)
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.tenancy import KeyLayout


def _tenant_id(value: str | None = None) -> str:
    return KeyLayout(value or os.environ.get("RAG_OPS_TENANT_ID", "default")).tenant_id


def _store(settings: Settings, tenant_id: str) -> TenantDynamoDbConfigStore:
    return TenantDynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
        tenant_id=tenant_id,
    )


def publish_config_revision(
    *,
    reason: str,
    actor: str,
    overrides: dict[str, object] | None = None,
    prefer_head: bool = False,
    tenant_id: str | None = None,
) -> PublishedRevision:
    tenant = _tenant_id(tenant_id)
    settings = Settings()
    entries = registry_entries()
    store = _store(settings, tenant)
    store.ensure_registry(entries)

    values = public_settings_values(settings)
    if prefer_head:
        head = store.get_head()
        if head is not None:
            published_values = store.get_revision_values(head.revision_no)
            if published_values:
                values.update(published_values)
    if overrides:
        values.update(overrides)

    for name, value in values.items():
        entry = entries.get(name)
        if entry is None or entry.sensitivity == "secret":
            raise ValueError(f"{name}: not publishable")
        validate_value_against_schema(entry, value)

    published = store.publish(values, actor=actor, change_reason=reason)
    snapshot = snapshot_for_values(settings, values, revision_no=published.revision_no)
    local_path = write_tenant_snapshot(snapshot, tenant)
    # Keep the active local snapshot for the CDK CONFIG_HASH bootstrap; tenant data remains
    # authoritative in the tenant partition and tenant-scoped S3 snapshot.
    write_local_snapshot(snapshot)
    try:
        upload_tenant_snapshot(snapshot, settings, tenant)
        snapshot_state = f"s3://{settings.s3_document_bucket}/{tenant_snapshot_key(tenant)}"
    except (BotoCoreError, ClientError, OSError) as exc:
        snapshot_state = f"unavailable ({type(exc).__name__})"

    print(
        f"CONFIG PUBLISHED tenant={tenant} revision={published.revision_no} "
        f"hash={published.content_hash} effective_hash={snapshot.config_hash} keys={len(values)}"
    )
    print(f"CONFIG SNAPSHOT local={local_path} s3={snapshot_state}")
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an append-only tenant configuration revision")
    parser.add_argument("--reason", required=True, help="Human change reason recorded in audit")
    parser.add_argument("--tenant-id", default=os.environ.get("RAG_OPS_TENANT_ID", "default"))
    parser.add_argument(
        "--actor",
        default=os.environ.get("USER") or os.environ.get("GITHUB_ACTOR") or "unknown",
    )
    args = parser.parse_args()

    publish_config_revision(reason=args.reason, actor=args.actor, tenant_id=args.tenant_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
