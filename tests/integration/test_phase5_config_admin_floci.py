from __future__ import annotations

import os
import uuid

import boto3
import pytest

from rag_ops_guard.configstore.admin import (
    AdminAuthorizationError,
    AdminPrincipal,
    ConfigAdminService,
    SecretMutationService,
)
from rag_ops_guard.configstore.admin_store import DynamoDbAdminStore
from rag_ops_guard.configstore.key_provider import KmsKeyProvider
from rag_ops_guard.configstore.phase3_secret_backend import EnvelopeSecretBackend
from rag_ops_guard.configstore.secret_service import EnvelopeSecretService
from rag_ops_guard.configstore.secret_store import DynamoDbSecretStore
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = "test"
SECRET_KEY = "test"


def _resource():
    return boto3.resource(
        "dynamodb",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def _kms():
    return boto3.client(
        "kms",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def _table():
    resource = _resource()
    name = f"rag-ops-admin-{uuid.uuid4().hex[:10]}"
    return resource.create_table(
        TableName=name,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _config_store(table_name: str, tenant_id: str) -> TenantDynamoDbConfigStore:
    return TenantDynamoDbConfigStore(
        endpoint_url=ENDPOINT,
        region=REGION,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        table=table_name,
        tenant_id=tenant_id,
    )


def _approval_store(table_name: str, tenant_id: str) -> DynamoDbAdminStore:
    return DynamoDbAdminStore(
        endpoint_url=ENDPOINT,
        region=REGION,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        table=table_name,
        tenant_id=tenant_id,
    )


def _principal(name: str, tenant_id: str) -> AdminPrincipal:
    return AdminPrincipal(principal_id=name, tenant_ids=frozenset({tenant_id}))


def _key_id(response: dict[str, object]) -> str:
    metadata = response.get("KeyMetadata")
    assert isinstance(metadata, dict)
    key_id = metadata.get("KeyId")
    assert isinstance(key_id, str)
    return key_id


def test_phase5_publish_rollback_and_dual_control_secret_flow_on_floci() -> None:
    table = _table()
    table_name = table.name
    kms = _kms()
    key_id = _key_id(kms.create_key(Description="phase5 config admin integration"))
    tenant_id = "tenant-a"
    operator_a = _principal("operator-a", tenant_id)
    operator_b = _principal("operator-b", tenant_id)

    try:
        config_admin = ConfigAdminService(
            store_factory=lambda tenant: _config_store(table_name, tenant)
        )
        first = config_admin.publish(
            principal=operator_a,
            tenant_id=tenant_id,
            values={"retrieval_top_k": 10},
            change_reason="phase5 baseline",
        )
        second = config_admin.publish(
            principal=operator_a,
            tenant_id=tenant_id,
            values={"retrieval_top_k": 20},
            change_reason="phase5 temporary change",
        )
        rolled = config_admin.rollback(
            principal=operator_a,
            tenant_id=tenant_id,
            revision_no=first.revision_no,
            change_reason="phase5 restore baseline",
        )

        store = _config_store(table_name, tenant_id)
        assert first.revision_no == 1
        assert second.revision_no == 2
        assert rolled.revision_no == 3
        head = store.get_head()
        assert head is not None and head.revision_no == 3
        assert store.get_revision_values(1) == {"retrieval_top_k": 10}
        assert store.get_revision_values(2) == {"retrieval_top_k": 20}
        assert store.get_revision_values(3) == {"retrieval_top_k": 10}
        audit = store.list_audit_events()
        assert audit[0].action == "rollback"
        assert audit[0].source_revision == 1
        assert audit[0].resulting_revision == 3

        provider = KmsKeyProvider(client=kms)
        envelope = EnvelopeSecretService(
            store=DynamoDbSecretStore(table=table),
            key_provider=provider,
        )
        secret_admin = SecretMutationService(
            store_factory=lambda tenant: _approval_store(table_name, tenant),
            backend_factory=lambda request: EnvelopeSecretBackend(
                service=envelope,
                kek_ref=request.kek_ref or "",
            ),
        )
        secret = b"phase5-integration-secret"
        request = secret_admin.request(
            principal=operator_a,
            tenant_id=tenant_id,
            action="set",
            key_name="external_api_key",
            change_reason="phase5 credential rotation",
            secret=secret,
            kek_ref=key_id,
        )
        persisted = _approval_store(table_name, tenant_id).get_secret_request(request.request_id)
        assert persisted is not None
        assert secret.decode() not in repr(persisted)

        with pytest.raises(AdminAuthorizationError):
            secret_admin.approve(
                principal=operator_a,
                tenant_id=tenant_id,
                request_id=request.request_id,
                secret=secret,
            )

        completed = secret_admin.approve(
            principal=operator_b,
            tenant_id=tenant_id,
            request_id=request.request_id,
            secret=secret,
        )
        assert completed.status == "completed"
        assert envelope.get_secret(
            scope="TENANT#tenant-a",
            key_name="external_api_key",
        ) == secret

        delete_request = secret_admin.request(
            principal=operator_a,
            tenant_id=tenant_id,
            action="delete",
            key_name="external_api_key",
            change_reason="credential retired",
        )
        secret_admin.approve(
            principal=operator_b,
            tenant_id=tenant_id,
            request_id=delete_request.request_id,
        )
        assert envelope.get_secret(
            scope="TENANT#tenant-a",
            key_name="external_api_key",
        ) is None

        raw_request = table.get_item(
            Key={"PK": "TENANT#tenant-a", "SK": f"SECRET_CHANGE#{request.request_id}"},
            ConsistentRead=True,
        )
        assert secret.decode() not in repr(raw_request)
        assert "payload_sha256" in repr(raw_request)
    finally:
        table.delete()
        try:
            kms.schedule_key_deletion(KeyId=key_id, PendingWindowInDays=7)
        except Exception:
            pass
