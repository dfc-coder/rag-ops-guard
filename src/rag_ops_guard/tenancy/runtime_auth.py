from __future__ import annotations

from functools import lru_cache
from typing import Any

from rag_ops_guard.runtime_settings import runtime_control_plane
from rag_ops_guard.tenancy.auth import TenantAuthenticator
from rag_ops_guard.tenancy.context import RequestContext
from rag_ops_guard.tenancy.dynamo_credentials import DynamoDbTenantCredentialStore


class MissingApiKeyError(ValueError):
    pass


def _header(event: dict[str, Any], name: str) -> str:
    raw = event.get("headers") or {}
    if not isinstance(raw, dict):
        return ""
    wanted = name.casefold()
    for key, value in raw.items():
        if str(key).casefold() == wanted and value is not None:
            return str(value).strip()
    return ""


@lru_cache(maxsize=1)
def _default_authenticator() -> TenantAuthenticator:
    control_plane = runtime_control_plane()
    store = DynamoDbTenantCredentialStore(
        table=control_plane.resources.tenant_credential_table,
    )
    return TenantAuthenticator(store=store)


def request_context_from_event(
    event: dict[str, Any],
    *,
    authenticator: TenantAuthenticator | Any | None = None,
) -> RequestContext:
    api_key = _header(event, "x-api-key")
    if not api_key:
        raise MissingApiKeyError("x-api-key header is required")
    resolved = authenticator or _default_authenticator()
    return resolved.authenticate(api_key)
