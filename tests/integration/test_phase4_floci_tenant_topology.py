from __future__ import annotations

import json
import os

import boto3
import pytest


@pytest.mark.integration
def test_floci_materializes_one_structural_vector_index_per_declared_tenant() -> None:
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
    region = os.environ.get("AWS_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "test")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")

    cloudformation = boto3.client(
        "cloudformation",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    stack = cloudformation.describe_stacks(StackName="RagOpsGuardLocal")["Stacks"][0]
    outputs = {
        item["OutputKey"]: item["OutputValue"]
        for item in stack.get("Outputs", [])
        if item.get("OutputKey") and item.get("OutputValue")
    }

    vector_bucket = outputs["VectorBucketName"]
    declared_indexes = json.loads(outputs["TenantVectorIndexNames"])
    assert isinstance(declared_indexes, list)
    assert declared_indexes
    assert all(isinstance(index_name, str) and index_name for index_name in declared_indexes)
    assert len(declared_indexes) == len(set(declared_indexes))

    vectors = boto3.client(
        "s3vectors",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    for index_name in declared_indexes:
        response = vectors.get_index(
            vectorBucketName=vector_bucket,
            indexName=index_name,
        )
        assert response["index"]["indexName"] == index_name
