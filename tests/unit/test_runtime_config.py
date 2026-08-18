from __future__ import annotations

from dataclasses import dataclass

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import ConfigHead
from rag_ops_guard.configstore.runtime import ConfigSnapshot, RuntimeConfigResolver


@dataclass
class FakeStore:
    values: dict[str, object]
    fail: bool = False
    head_calls: int = 0
    value_calls: int = 0

    def get_head(self) -> ConfigHead | None:
        self.head_calls += 1
        if self.fail:
            raise RuntimeError("dynamodb unavailable")
        return ConfigHead(revision_no=3, content_hash="db-head-hash")

    def get_revision_values(self, revision_no: int) -> dict[str, object]:
        assert revision_no == 3
        self.value_calls += 1
        if self.fail:
            raise RuntimeError("dynamodb unavailable")
        return dict(self.values)


def test_db_value_wins_over_environment_and_hash_is_effective() -> None:
    now = [0.0]
    store = FakeStore({"retrieval_top_k": 9, "retrieval_context_k": 3})
    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None, retrieval_top_k=2, retrieval_context_k=2),
        source="db",
        clock=lambda: now[0],
    )

    effective = resolver.resolve()

    assert effective.settings.retrieval_top_k == 9
    assert effective.settings.retrieval_context_k == 3
    assert effective.source == "dynamodb"
    assert len(effective.config_hash) == 64
    assert not effective.stale
    assert not effective.fail_closed


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


def test_dynamodb_outage_serves_stale_tuning_then_fails_closed_after_max_stale() -> None:
    now = [0.0]
    store = FakeStore(
        {
            "retrieval_top_k": 7,
            "retrieval_domain_min_relevance": 0.6,
            "retrieval_min_relevance": 0.7,
        }
    )
    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        head_ttl_s=30.0,
        max_stale_s=120.0,
        clock=lambda: now[0],
    )
    resolver.resolve()

    store.fail = True
    now[0] = 40.0
    stale = resolver.resolve()
    assert stale.settings.retrieval_top_k == 7
    assert stale.stale
    assert stale.source == "cache"
    assert not stale.fail_closed

    now[0] = 150.0
    expired = resolver.resolve()
    assert expired.stale
    assert expired.fail_closed


def test_fallback_order_is_s3_then_baked_then_code_default() -> None:
    now = [0.0]
    store = FakeStore({}, fail=True)
    s3_snapshot = ConfigSnapshot(
        revision_no=2,
        config_hash="s3-hash",
        values={"retrieval_top_k": 11},
        created_at=0.0,
    )
    baked_snapshot = ConfigSnapshot(
        revision_no=1,
        config_hash="baked-hash",
        values={"retrieval_top_k": 12},
        created_at=0.0,
    )

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        clock=lambda: now[0],
        s3_loader=lambda: s3_snapshot,
        baked_loader=lambda: baked_snapshot,
    )
    assert resolver.resolve().source == "s3_snapshot"
    assert resolver.resolve().settings.retrieval_top_k == 11

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        clock=lambda: now[0],
        s3_loader=lambda: None,
        baked_loader=lambda: baked_snapshot,
    )
    assert resolver.resolve().source == "baked"
    assert resolver.resolve().settings.retrieval_top_k == 12

    resolver = RuntimeConfigResolver(
        store=store,
        bootstrap=Settings(_env_file=None),
        source="db",
        clock=lambda: now[0],
        s3_loader=lambda: None,
        baked_loader=lambda: None,
    )
    fallback = resolver.resolve()
    assert fallback.source == "code_default"
    assert fallback.settings.retrieval_top_k == 20
