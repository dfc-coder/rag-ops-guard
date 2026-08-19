"""SPEC-P4-MULTITENANCY 4.5 red contracts for tenant-scoped config records."""

from typing import Any

import pytest

from rag_ops_guard.configstore import tenant_store
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.tenancy.config_scope import tenant_config_scope


class FakeDynamoClient:
    def __init__(self) -> None:
        self.get_keys: list[dict[str, Any]] = []

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.get_keys.append(kwargs["Key"])
        return {}


def test_tenant_config_scope_is_deterministic_and_validated() -> None:
    assert tenant_config_scope("tenant-a") == "TENANT#tenant-a"
    assert tenant_config_scope("tenant-b") == "TENANT#tenant-b"
    with pytest.raises(ValueError):
        tenant_config_scope("../tenant-a")


def test_config_head_lookup_uses_tenant_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDynamoClient()
    monkeypatch.setattr(tenant_store.boto3, "client", lambda *args, **kwargs: fake)

    store_a = TenantDynamoDbConfigStore(
        endpoint_url="http://floci",
        region="us-east-1",
        access_key="test",
        secret_key="test",
        table="rag-ops-config",
        tenant_id="tenant-a",
    )
    store_b = TenantDynamoDbConfigStore(
        endpoint_url="http://floci",
        region="us-east-1",
        access_key="test",
        secret_key="test",
        table="rag-ops-config",
        tenant_id="tenant-b",
    )

    assert store_a.get_head() is None
    assert store_b.get_head() is None
    assert fake.get_keys[0]["PK"] == {"S": "TENANT#tenant-a"}
    assert fake.get_keys[1]["PK"] == {"S": "TENANT#tenant-b"}


def test_scope_is_not_supplied_by_request_payload() -> None:
    assert tenant_config_scope("tenant-a") != tenant_config_scope("tenant-b")
