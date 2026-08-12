from __future__ import annotations

import os
import uuid

import boto3
import pytest
from botocore.config import Config

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def client(service: str, **kwargs: object):
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        **kwargs,
    )


def test_s3_roundtrip() -> None:
    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    bucket = f"rag-ops-ci-{uuid.uuid4().hex[:12]}"
    s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key="raw/test.md", Body=b"hello")
    response = s3.get_object(Bucket=bucket, Key="raw/test.md")
    assert response["Body"].read() == b"hello"


def test_s3_vectors_put_query_delete() -> None:
    vectors = client("s3vectors")
    bucket = f"rag-ops-vectors-{uuid.uuid4().hex[:10]}"
    index = "ops-knowledge-v1"
    vectors.create_vector_bucket(vectorBucketName=bucket)
    vectors.create_index(
        vectorBucketName=bucket,
        indexName=index,
        dataType="float32",
        dimension=4,
        distanceMetric="cosine",
    )
    vectors.put_vectors(
        vectorBucketName=bucket,
        indexName=index,
        vectors=[
            {"key": "a", "data": {"float32": [1.0, 0.0, 0.0, 0.0]}, "metadata": {"kind": "a"}},
            {"key": "b", "data": {"float32": [0.0, 1.0, 0.0, 0.0]}, "metadata": {"kind": "b"}},
        ],
    )
    response = vectors.query_vectors(
        vectorBucketName=bucket,
        indexName=index,
        queryVector={"float32": [0.99, 0.01, 0.0, 0.0]},
        topK=1,
        returnMetadata=True,
        returnDistance=True,
    )
    assert response["vectors"][0]["key"] == "a"
    assert response["vectors"][0]["metadata"]["kind"] == "a"
    vectors.delete_vectors(vectorBucketName=bucket, indexName=index, keys=["a"])
    remaining = vectors.get_vectors(vectorBucketName=bucket, indexName=index, keys=["a"])
    assert remaining.get("vectors", []) == []
