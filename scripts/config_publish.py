from __future__ import annotations

import argparse
import os

from rag_ops_guard.config import get_settings
from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore
from rag_ops_guard.configstore.registry import (
    public_settings_values,
    registry_entries,
    validate_value_against_schema,
)


def _store() -> DynamoDbConfigStore:
    settings = get_settings()
    return DynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an append-only configuration revision")
    parser.add_argument("--reason", required=True, help="Human change reason recorded in audit")
    parser.add_argument(
        "--actor",
        default=os.environ.get("USER") or os.environ.get("GITHUB_ACTOR") or "unknown",
    )
    args = parser.parse_args()

    settings = get_settings()
    entries = registry_entries()
    values = public_settings_values(settings)
    for name, value in values.items():
        validate_value_against_schema(entries[name], value)

    store = _store()
    store.ensure_registry(entries)
    published = store.publish(values, actor=args.actor, change_reason=args.reason)
    print(
        f"CONFIG PUBLISHED revision={published.revision_no} "
        f"hash={published.content_hash} keys={len(values)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
