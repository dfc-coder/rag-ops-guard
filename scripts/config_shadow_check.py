from __future__ import annotations

import json
import os

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore
from rag_ops_guard.configstore.registry import public_settings_values
from rag_ops_guard.configstore.resolver import resolve_shadow


def _store(settings: Settings) -> DynamoDbConfigStore:
    return DynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
    )


def _display(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main() -> int:
    # Shadow mode intentionally compares the publisher-facing environment with the DB revision;
    # do not use the effective resolver here or divergence would compare DB with itself.
    settings = Settings()
    local_values = public_settings_values(settings)
    result = resolve_shadow(_store(settings), local_values)

    if result.revision_no is None:
        print("CONFIG SHADOW DIVERGENCE: no published HEAD")
        print(f"local_hash={result.local_hash}")
        return 1

    print(
        f"CONFIG SHADOW revision={result.revision_no} "
        f"local_hash={result.local_hash} db_hash={result.db_hash}"
    )
    for key in result.divergences:
        print(
            f"DIVERGENCE key={key} env_value={_display(local_values.get(key))} "
            f"db_value={_display(result.db_values.get(key))}"
        )

    db_recomputed = None
    if result.db_values:
        from rag_ops_guard.configstore.hashing import content_hash

        db_recomputed = content_hash(result.db_values)
    hash_mismatch = db_recomputed is not None and db_recomputed != result.db_hash
    if hash_mismatch:
        print(f"DIVERGENCE db_head_hash={result.db_hash} db_recomputed_hash={db_recomputed}")

    if result.divergences or result.local_hash != result.db_hash or hash_mismatch:
        print("CONFIG SHADOW: DIVERGED")
        return 1
    print("CONFIG SHADOW: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
