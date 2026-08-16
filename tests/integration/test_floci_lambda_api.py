from __future__ import annotations

import io
import json
import os
import uuid
import zipfile
from contextlib import suppress
from typing import Any

import boto3
import httpx
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def client(service: str, **kwargs: object) -> Any:
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        **kwargs,
    )


def lambda_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "index.py",
            "def handler(event, context):\n"
            "    del context\n"
            "    return {'statusCode': 200, 'headers': {'content-type': 'application/json'}, "
            "'body': '{\"status\":\"floci-ok\"}'}\n",
        )
    return buffer.getvalue()


def test_floci_standard_s3_lambda_and_http_api_data_plane() -> None:
    suffix = uuid.uuid4().hex[:10]
    bucket = f"rag-ops-code-{suffix}"
    key = "lambda/function.zip"
    function_name = f"rag-ops-ci-{suffix}"
    api_id: str | None = None

    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    lamb = client("lambda")
    api = client("apigatewayv2")

    try:
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=lambda_zip(), ContentType="application/zip")

        function = lamb.create_function(
            FunctionName=function_name,
            Runtime="python3.12",
            Role="arn:aws:iam::000000000000:role/rag-ops-ci",
            Handler="index.handler",
            Code={"S3Bucket": bucket, "S3Key": key},
            Timeout=30,
            MemorySize=256,
        )

        direct = lamb.invoke(
            FunctionName=function_name,
            Payload=json.dumps({"source": "integration-test"}).encode(),
        )
        assert direct.get("FunctionError") is None
        direct_payload = json.loads(direct["Payload"].read())
        assert direct_payload["statusCode"] == 200
        assert json.loads(direct_payload["body"])["status"] == "floci-ok"

        created_api = api.create_api(
            Name=f"rag-ops-ci-{suffix}",
            ProtocolType="HTTP",
        )
        api_id = str(created_api["ApiId"])
        integration = api.create_integration(
            ApiId=api_id,
            IntegrationType="AWS_PROXY",
            IntegrationUri=function["FunctionArn"],
            PayloadFormatVersion="2.0",
        )
        api.create_route(
            ApiId=api_id,
            RouteKey="GET /health",
            Target=f"integrations/{integration['IntegrationId']}",
        )
        api.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)

        endpoint = f"{ENDPOINT.rstrip('/')}/execute-api/{api_id}/$default/health"
        response = httpx.get(endpoint, timeout=30)
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "floci-ok"}
    finally:
        if api_id is not None:
            with suppress(ClientError):
                api.delete_api(ApiId=api_id)
        with suppress(ClientError):
            lamb.delete_function(FunctionName=function_name)
        with suppress(ClientError):
            s3.delete_object(Bucket=bucket, Key=key)
        with suppress(ClientError):
            s3.delete_bucket(Bucket=bucket)
