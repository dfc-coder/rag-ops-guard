from __future__ import annotations

from dataclasses import dataclass

from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore
from rag_ops_guard.configstore.hashing import content_hash


@dataclass(frozen=True)
class ShadowResolution:
    revision_no: int | None
    db_hash: str | None
    local_hash: str
    db_values: dict[str, object]
    divergences: tuple[str, ...]


def resolve_shadow(
    store: DynamoDbConfigStore,
    local_values: dict[str, object],
) -> ShadowResolution:
    local_digest = content_hash(local_values)
    head = store.get_head()
    if head is None:
        return ShadowResolution(
            revision_no=None,
            db_hash=None,
            local_hash=local_digest,
            db_values={},
            divergences=("<missing_head>",),
        )
    db_values = store.get_revision_values(head.revision_no)
    keys = sorted(set(local_values) | set(db_values))
    divergences = tuple(key for key in keys if local_values.get(key) != db_values.get(key))
    return ShadowResolution(
        revision_no=head.revision_no,
        db_hash=head.content_hash,
        local_hash=local_digest,
        db_values=db_values,
        divergences=divergences,
    )
