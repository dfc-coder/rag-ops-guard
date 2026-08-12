from __future__ import annotations

import io
import json
import os
import uuid
import zipfile
from typing import Any

import boto3
import pytest

pytestmark = pytest.mark.integration

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def client(service: str) -> Any:
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def lambda_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "index.py",
            "def handler(event, context):\n"
            "    del context\n"
            "    return {'statusCode': 200, 'body': 'floci-ok', 'event': event}\n",
        )
    return buffer.getvalue()


def test_floci_executes_lambda_and_accepts_http_api_lambda_integration() -> None:
    suffix = uuid.uuid4().hex[:8]
    function_name = f"rag-ops-ci-{suffix}"
    lamb = client("lambda")
    api = client("apigatewayv2")

    function = lamb.create_function(
        FunctionName=function_name,
        Runtime="python3.12",
        Role="arn:aws:iam::000000000000:role/rag-ops-ci",
        Handler="index.handler",
        Code={"ZipFile": lambda_zip()},
        Timeout=30,
        MemorySize=256,
    )
    response = lamb.invoke(
        FunctionName=function_name,
        Payload=json.dumps({"source": "integration-test"}).encode(),
    )
    payload = json.loads(response["Payload"].read())
    assert payload["statusCode"] == 200
    assert payload["body"] == "floci-ok"
    assert payload["event"]["source"] == "integration-test"

    created_api = api.create_api(
        Name=f"rag-ops-ci-{suffix}",
        ProtocolType="HTTP",
        Tags={"floci:override-id": f"ragops{suffix}"},
    )
    integration = api.create_integration(
        ApiId=created_api["ApiId"],
        IntegrationType="AWS_PROXY",
        IntegrationUri=function["FunctionArn"],
        PayloadFormatVersion="2.0",
    )
    route = api.create_route(
        ApiId=created_api["ApiId"],
        RouteKey="GET /health",
        Target=f"integrations/{integration['IntegrationId']}",
    )
    api.create_stage(ApiId=created_api["ApiId"], StageName="$default", AutoDeploy=True)

    routes = api.get_routes(ApiId=created_api["ApiId"])["Items"]
    assert any(item["RouteId"] == route["RouteId"] for item in routes)
    assert any(item["RouteKey"] == "GET /health" for item in routes)
