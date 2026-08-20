from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rag_ops_guard.tenancy.aws_session import tenant_session
from rag_ops_guard.tenancy.context import RequestContext


class FakeSts:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def assume_role(self, **kwargs: object) -> dict[str, object]:
        self.requests.append(dict(kwargs))
        return {
            "Credentials": {
                "AccessKeyId": "ASIAEXAMPLE",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
                "Expiration": datetime.now(UTC) + timedelta(hours=1),
            }
        }


def test_tenant_session_tag_is_derived_only_from_authenticated_context() -> None:
    sts = FakeSts()
    context = RequestContext(principal="api-key:key-a", tenant_id="tenant-a")

    session = tenant_session(
        context,
        role_arn="arn:aws:iam::123456789012:role/rag-ops-guard-tenant-data",
        sts_client=sts,
    )

    assert session is not None
    assert sts.requests == [
        {
            "RoleArn": "arn:aws:iam::123456789012:role/rag-ops-guard-tenant-data",
            "RoleSessionName": "rag-ops-tenant-a",
            "Tags": [{"Key": "tenant_id", "Value": "tenant-a"}],
            "TransitiveTagKeys": ["tenant_id"],
        }
    ]
