from __future__ import annotations

import os

import boto3
import pytest


@pytest.mark.integration
def test_floci_materializes_one_structural_vector_index_per_declared_tenant() -> None:
    client = boto3.client(
        "s3vectors",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )

    for tenant_id in ("default", "tenant-b"):
        response = client.get_index(
            vectorBucketName="rag-ops-guard-vectors-local",
            indexName=f"ops-knowledge-openvino-v1--{tenant_id}",
        )
        assert response["index"]["indexName"] == f"ops-knowledge-openvino-v1--{tenant_id}"
