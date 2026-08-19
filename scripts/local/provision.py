from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
STACK_NAME = os.environ.get("RAG_OPS_CDK_STACK_NAME", "RagOpsGuardLocal")
DEFAULT_API_STAGE = "$default"


def client(service: str, **kwargs: object) -> Any:
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **kwargs,
    )


def stack_outputs() -> dict[str, str]:
    response = client("cloudformation").describe_stacks(StackName=STACK_NAME)
    stacks = response.get("Stacks", [])
    if len(stacks) != 1:
        raise RuntimeError(f"expected one CloudFormation stack named {STACK_NAME!r}")
    return {
        str(item["OutputKey"]): str(item["OutputValue"])
        for item in stacks[0].get("Outputs", [])
        if item.get("OutputKey") and item.get("OutputValue")
    }


def required_output(outputs: dict[str, str], name: str) -> str:
    value = outputs.get(name, "").strip()
    if not value:
        raise RuntimeError(f"CDK stack output {name!r} is missing")
    return value


def materialize_floci_s3_vectors(outputs: dict[str, str]) -> None:
    """Bridge the one IaC gap in Floci 1.6.0.

    Floci exposes the S3 Vectors data-plane API but its CloudFormation engine does not
    materialize AWS::S3Vectors::* resources yet. CDK remains the source of names and
    dimensions; this adapter only realizes those declared outputs through the same AWS API.
    """

    bucket = required_output(outputs, "VectorBucketName")
    index = required_output(outputs, "VectorIndexName")
    dimension = int(required_output(outputs, "VectorDimension"))
    vectors = client("s3vectors")

    try:
        vectors.get_vector_bucket(vectorBucketName=bucket)
    except ClientError:
        vectors.create_vector_bucket(vectorBucketName=bucket)

    try:
        vectors.get_index(vectorBucketName=bucket, indexName=index)
    except ClientError:
        vectors.create_index(
            vectorBucketName=bucket,
            indexName=index,
            dataType="float32",
            dimension=dimension,
            distanceMetric="cosine",
        )


def parse_proxy_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Lambda returned non-JSON payload: {raw[:1000]!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Lambda returned unexpected payload: {payload!r}")
    return payload


def validate_invalid_request_payload(payload: dict[str, Any], *, source: str) -> None:
    if payload.get("statusCode") != 400:
        raise RuntimeError(f"{source} expected HTTP 400 proxy response, got {payload!r}")
    body = payload.get("body")
    try:
        parsed_body = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source} returned invalid JSON body: {body!r}") from exc
    if not isinstance(parsed_body, dict) or parsed_body.get("error") != "invalid_request":
        raise RuntimeError(f"{source} did not reach the expected handler: {payload!r}")


def probe_lambda(function_name: str) -> None:
    response = client("lambda").invoke(FunctionName=function_name, Payload=b"{}")
    raw = response["Payload"].read()
    function_error = response.get("FunctionError")
    if function_error:
        raise RuntimeError(
            f"Lambda {function_name} failed during direct invocation "
            f"({function_error}): {raw[:2000].decode('utf-8', errors='replace')}"
        )
    validate_invalid_request_payload(
        parse_proxy_payload(raw),
        source=f"Lambda {function_name}",
    )


def floci_execution_endpoint(base_endpoint: str, api_id: str) -> str:
    return f"{base_endpoint.rstrip('/')}/execute-api/{api_id}/{DEFAULT_API_STAGE}"


def validate_api_probe(response: httpx.Response, *, route: str) -> None:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if (
        response.status_code == 400
        and isinstance(payload, dict)
        and payload.get("error") == "invalid_request"
    ):
        return
    content_type = response.headers.get("content-type", "unknown")
    raise RuntimeError(
        f"API Gateway data-plane probe failed for {route}: expected HTTP 400 JSON "
        f"invalid_request, got HTTP {response.status_code} ({content_type}): "
        f"{response.text[:1000]}"
    )


def probe_api_route(endpoint: str, route: str) -> None:
    response = httpx.post(f"{endpoint}{route}", json={}, timeout=120)
    validate_api_probe(response, route=route)


def main() -> None:
    outputs = stack_outputs()
    materialize_floci_s3_vectors(outputs)

    ingest_name = required_output(outputs, "IngestFunctionName")
    query_name = required_output(outputs, "QueryFunctionName")
    api_id = required_output(outputs, "ApiId")

    probe_lambda(ingest_name)
    probe_lambda(query_name)

    endpoint = floci_execution_endpoint(ENDPOINT, api_id)
    probe_api_route(endpoint, "/v1/ingest")
    probe_api_route(endpoint, "/v1/query")

    Path(".local").mkdir(exist_ok=True)
    Path(".local/api-url").write_text(endpoint, encoding="utf-8")

    print(f"CDK stack: {STACK_NAME}")
    print("Floci S3 Vectors bridge: ready")
    print("Lambda direct invoke: ready")
    print(f"Local API: {endpoint}")
    print("API data plane: ready")


if __name__ == "__main__":
    main()
