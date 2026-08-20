from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
import botocore.session
from botocore.credentials import RefreshableCredentials

from rag_ops_guard.tenancy.context import RequestContext


def _session_name(tenant_id: str) -> str:
    return f"rag-ops-{tenant_id}"[:64]


def _credential_metadata(response: dict[str, Any]) -> dict[str, str]:
    credentials = response["Credentials"]
    expiration = credentials["Expiration"]
    expiry_time = expiration.isoformat() if hasattr(expiration, "isoformat") else str(expiration)
    return {
        "access_key": str(credentials["AccessKeyId"]),
        "secret_key": str(credentials["SecretAccessKey"]),
        "token": str(credentials["SessionToken"]),
        "expiry_time": expiry_time,
    }


def tenant_session(
    context: RequestContext,
    *,
    role_arn: str,
    sts_client: Any | None = None,
) -> boto3.Session:
    """Assume the tenant data role using only the authenticated RequestContext tenant id."""

    trusted_tenant = context.tenant_id
    if not trusted_tenant:
        raise ValueError("authenticated tenant_id is required")
    if not role_arn.strip():
        raise ValueError("tenant data role ARN is required")
    sts = sts_client or boto3.client("sts")

    def refresh() -> dict[str, str]:
        response = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=_session_name(trusted_tenant),
            Tags=[{"Key": "tenant_id", "Value": trusted_tenant}],
            TransitiveTagKeys=["tenant_id"],
        )
        return _credential_metadata(response)

    credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method="sts-assume-role-tenant",
    )
    botocore_session = botocore.session.get_session()
    botocore_session._credentials = credentials
    region = boto3.Session().region_name
    if region:
        botocore_session.set_config_variable("region", region)
    return boto3.Session(botocore_session=botocore_session)
