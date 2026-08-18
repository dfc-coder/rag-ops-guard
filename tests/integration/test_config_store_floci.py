from __future__ import annotations

import os
import uuid

import boto3
import pytest
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
SERIALIZER = TypeSerializer()


def _client():
    return boto3.client(
        "dynamodb",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _table() -> str:
    name = f"rag-ops-config-{uuid.uuid4().hex[:10]}"
    _client().create_table(
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
    return name


def _store(table: str) -> DynamoDbConfigStore:
    return DynamoDbConfigStore(
        endpoint_url=ENDPOINT,
        region=REGION,
        access_key="test",
        secret_key="test",
        table=table,
    )


def test_publish_is_append_only_and_head_monotonic() -> None:
    table = _table()
    store = _store(table)
    values = {"retrieval_top_k": 20, "llm_temperature": 0.7}

    first = store.publish(values, actor="integration", change_reason="baseline")
    head = store.get_head()
    assert head is not None
    assert head.revision_no == 1
    assert head.content_hash == first.content_hash
    assert store.get_revision_values(1) == values

    second = store.publish(values, actor="integration", change_reason="same-values")
    assert second.revision_no == 2
    assert second.content_hash == first.content_hash
    head = store.get_head()
    assert head is not None and head.revision_no == 2

    serializer = SERIALIZER
    with pytest.raises(ClientError) as existing_value:
        _client().put_item(
            TableName=table,
            Item={
                "PK": serializer.serialize("GLOBAL"),
                "SK": serializer.serialize("VAL#retrieval_top_k#00000000000000000001"),
                "value_json": serializer.serialize("21"),
            },
            ConditionExpression="attribute_not_exists(SK)",
        )
    assert existing_value.value.response["Error"]["Code"] == "ConditionalCheckFailedException"

    with pytest.raises(ClientError):
        _client().put_item(
            TableName=table,
            Item={
                "PK": serializer.serialize("GLOBAL"),
                "SK": serializer.serialize("HEAD"),
                "revision_no": serializer.serialize(1),
                "content_hash": serializer.serialize("0" * 64),
            },
            ConditionExpression="revision_no < :n",
            ExpressionAttributeValues={":n": serializer.serialize(1)},
        )
    head = store.get_head()
    assert head is not None and head.revision_no == 2
