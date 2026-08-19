from __future__ import annotations

from rag_ops_guard.tenancy.key_layout import KeyLayout


def tenant_config_scope(tenant_id: str) -> str:
    """Canonical DynamoDB partition key for one tenant's effective config history."""
    return f"TENANT#{KeyLayout(tenant_id).tenant_id}"
