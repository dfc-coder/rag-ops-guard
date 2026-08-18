from __future__ import annotations

from dataclasses import dataclass

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import ConfigHead
from rag_ops_guard.configstore.runtime import (
    RuntimeConfigResolver,
    snapshot_for_values,
)


@dataclass
class FakeStore:
    values: dict[str, object]
    fail: bool = False
    head_calls: int = 0
    value_calls: int = 0
    published_at: float | None = None

    def get_head(self) -> ConfigHead | None:
        self.head_calls += 1
        if self.fail:
            raise RuntimeError("dynamodb unavailable")
        return ConfigHead(
            revision_no=3,
            content_hash="db-head-hash",
            published_at=self.published_at,
        )

    def get_revision_values(self, revision_no: int) -> dict[str, object]:
        assert revision_no == 3
        self.value_calls += 1
        if self.fail:
            raise RuntimeError("dynamodb unavailable")
        return dict(self.values)


def test_db_value_wins_over_environment_and_hash_is_effective() -> None:
    now = [100.0]
    store = FakeStore(
        {"retrieval_top_k": 9, "retrieval_context_k": 3},
        published_at=88.0,
    )
    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None, retrieval_top_k=2, retrieval_context_k=2),
        source="db",
        clock=lambda: now[0],
        wall_clock=lambda: now[0],
    )

    effective = resolver.resolve()

    assert effective.settings.retrieval_top_k == 9
    assert effective.settings.retrieval_context_k == 3
    assert effective.source == "dynamodb"
    assert len(effective.config_hash) == 64
    assert not effective.stale
    assert not effective.fail_closed
    assert not effective.db_unavailable
    assert effective.revision_age_s == 12.0


def test_l1_cache_is_revision_keyed_and_head_is_checked_only_after_ttl() -> None:
    now = [0.0]
    store = FakeStore({"retrieval_top_k": 8})
    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        head_ttl_s=45.0,
        clock=lambda: now[0],
    )

    first = resolver.resolve()
    now[0] = 10.0
    second = resolver.resolve()
    now[0] = 46.0
    third = resolver.resolve()

    assert first.config_hash == second.config_hash == third.config_hash
    assert store.head_calls == 2
    assert store.value_calls == 1
    assert second.source == "cache"
    assert not second.db_unavailable


def test_dynamodb_outage_serves_stale_tuning_then_fails_closed_after_max_stale() -> None:
    now = [0.0]
    store = FakeStore(
        {
            "retrieval_top_k": 7,
            "retrieval_domain_min_relevance": 0.6,
            "retrieval_min_relevance": 0.7,
        },
        published_at=0.0,
    )
    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        head_ttl_s=30.0,
        max_stale_s=120.0,
        clock=lambda: now[0],
        wall_clock=lambda: now[0],
    )
    resolver.resolve()

    store.fail = True
    now[0] = 40.0
    stale = resolver.resolve()
    assert stale.settings.retrieval_top_k == 7
    assert stale.stale
    assert stale.source == "cache"
    assert stale.db_unavailable
    assert stale.revision_age_s == 40.0
    assert not stale.fail_closed

    now[0] = 150.0
    expired = resolver.resolve()
    assert expired.stale
    assert expired.db_unavailable
    assert expired.revision_age_s == 150.0
    assert expired.fail_closed


def test_fallback_order_is_s3_then_baked_then_code_default() -> None:
    now = [50.0]
    bootstrap = Settings(_env_file=None)
    store = FakeStore({}, fail=True)
    s3_snapshot = snapshot_for_values(
        bootstrap,
        {"retrieval_top_k": 11},
        revision_no=2,
        created_at=10.0,
    )
    baked_snapshot = snapshot_for_values(
        bootstrap,
        {"retrieval_top_k": 12},
        revision_no=1,
        created_at=20.0,
    )

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=bootstrap,
        source="db",
        clock=lambda: now[0],
        wall_clock=lambda: now[0],
        s3_loader=lambda: s3_snapshot,
        baked_loader=lambda: baked_snapshot,
    )
    s3 = resolver.resolve()
    assert s3.source == "s3_snapshot"
    assert s3.settings.retrieval_top_k == 11
    assert s3.db_unavailable
    assert s3.revision_age_s == 40.0

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=bootstrap,
        source="db",
        clock=lambda: now[0],
        wall_clock=lambda: now[0],
        s3_loader=lambda: None,
        baked_loader=lambda: baked_snapshot,
    )
    baked = resolver.resolve()
    assert baked.source == "baked"
    assert baked.settings.retrieval_top_k == 12
    assert baked.db_unavailable
    assert baked.revision_age_s == 30.0

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=bootstrap,
        source="db",
        clock=lambda: now[0],
        wall_clock=lambda: now[0],
        s3_loader=lambda: None,
        baked_loader=lambda: None,
    )
    fallback = resolver.resolve()
    assert fallback.source == "code_default"
    assert fallback.settings.retrieval_top_k == 20
    assert fallback.db_unavailable
    assert fallback.revision_age_s is None
