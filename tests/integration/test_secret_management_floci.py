from __future__ import annotations

import os
import uuid

import boto3
import pytest

from rag_ops_guard.configstore.key_provider import KmsKeyProvider
from rag_ops_guard.configstore.secret_service import EnvelopeSecretService
from rag_ops_guard.configstore.secret_store import DynamoDbSecretStore

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def _resource():
    return boto3.resource(
        "dynamodb",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _kms():
    return boto3.client(
        "kms",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _table():
    resource = _resource()
    name = f"rag-ops-secrets-{uuid.uuid4().hex[:10]}"
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


def _key_id(response: dict[str, object]) -> str:
    metadata = response.get("KeyMetadata")
    assert isinstance(metadata, dict)
    key_id = metadata.get("KeyId")
    assert isinstance(key_id, str)
    return key_id


def test_floci_secret_envelope_dek_and_rotation_contract() -> None:
    table = _table()
    kms = _kms()
    key_ids: list[str] = []
    try:
        kek_v1 = _key_id(kms.create_key(Description="phase3 integration kek-v1"))
        kek_v2 = _key_id(kms.create_key(Description="phase3 integration kek-v2"))
        key_ids.extend((kek_v1, kek_v2))
        provider = KmsKeyProvider(client=kms)
        store = DynamoDbSecretStore(table=table)
        service = EnvelopeSecretService(store=store, key_provider=provider)

        service.set_secret(
            scope="tenant-a",
            key_name="openai_api_key",
            secret=b"integration-secret",
            kek_ref=kek_v1,
        )
        before = store.get_secret(scope="tenant-a", key_name="openai_api_key")
        assert before is not None
        assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"integration-secret"

        assert service.rotate_kek(scope="tenant-a", new_kek_ref=kek_v2) == 1
        after_kek = store.get_secret(scope="tenant-a", key_name="openai_api_key")
        assert after_kek == before
        assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"integration-secret"

        assert service.rotate_dek(scope="tenant-a", kek_ref=kek_v2) == 2
        after_dek = store.get_secret(scope="tenant-a", key_name="openai_api_key")
        assert after_dek is not None
        assert after_dek.dek_version == 2
        assert after_dek.ciphertext != before.ciphertext
        assert service.get_secret(scope="tenant-a", key_name="openai_api_key") == b"integration-secret"
    finally:
        table.delete()
        for key_id in key_ids:
            try:
                kms.schedule_key_deletion(KeyId=key_id, PendingWindowInDays=7)
            except Exception:
                pass
