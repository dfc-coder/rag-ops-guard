from __future__ import annotations

import os
from typing import Any

from rag_ops_guard.config import Settings
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


def _default_authenticator() -> TenantAuthenticator:
    settings = Settings()
    store = DynamoDbTenantCredentialStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("TENANT_TABLE", "rag-ops-tenants"),
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
