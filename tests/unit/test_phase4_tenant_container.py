"""SPEC-P4-MULTITENANCY 4.2 red contracts for tenant/config dependency isolation."""

from rag_ops_guard.container import Container
from rag_ops_guard.tenancy import RequestContext


def _ctx(tenant_id: str) -> RequestContext:
    return RequestContext(principal=f"api-key:{tenant_id}", tenant_id=tenant_id)


def test_container_reuses_only_same_tenant_and_config_generation() -> None:
    container: Container[object] = Container(max_generations=32)
    created: list[tuple[str, str]] = []

    def factory(context: RequestContext, config_hash: str) -> object:
        created.append((context.tenant_id, config_hash))
        return object()

    a1 = container.get(_ctx("tenant-a"), "hash-1", factory)
    a1_again = container.get(_ctx("tenant-a"), "hash-1", factory)
    b1 = container.get(_ctx("tenant-b"), "hash-1", factory)
    a2 = container.get(_ctx("tenant-a"), "hash-2", factory)

    assert a1 is a1_again
    assert a1 is not b1
    assert a1 is not a2
    assert created == [
        ("tenant-a", "hash-1"),
        ("tenant-b", "hash-1"),
        ("tenant-a", "hash-2"),
    ]


def test_container_is_lru_bounded_to_32_tenant_config_generations() -> None:
    container: Container[object] = Container(max_generations=32)
    context = _ctx("tenant-a")

    def factory(_context: RequestContext, _config_hash: str) -> object:
        return object()

    first = container.get(context, "hash-00", factory)
    for index in range(1, 32):
        container.get(context, f"hash-{index:02d}", factory)

    # Touch the first generation so hash-01 becomes the least-recently-used entry.
    assert container.get(context, "hash-00", factory) is first
    container.get(context, "hash-32", factory)

    assert len(container) == 32
    assert ("tenant-a", "hash-00") in container.keys()
    assert ("tenant-a", "hash-01") not in container.keys()


def test_same_config_hash_never_shares_warm_dependency_between_tenants() -> None:
    container: Container[dict[str, str]] = Container(max_generations=32)

    def factory(context: RequestContext, config_hash: str) -> dict[str, str]:
        return {"tenant": context.tenant_id, "config": config_hash}

    tenant_a = container.get(_ctx("tenant-a"), "same-hash", factory)
    tenant_b = container.get(_ctx("tenant-b"), "same-hash", factory)

    assert tenant_a == {"tenant": "tenant-a", "config": "same-hash"}
    assert tenant_b == {"tenant": "tenant-b", "config": "same-hash"}
    assert tenant_a is not tenant_b
