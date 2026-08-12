from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError


def _client(service: str) -> Any:
    return boto3.client(
        service,
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _lambda_exists(client: Any, name: str) -> bool:
    try:
        client.get_function(FunctionName=name)
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404 or exc.response.get("Error", {}).get("Code") in {
            "ResourceNotFoundException",
            "404",
        }:
            return False
        raise


def main() -> None:
    api = _client("apigatewayv2")
    for existing in api.get_apis().get("Items", []):
        if existing.get("Name") == "rag-ops-guard-local":
            api.delete_api(ApiId=existing["ApiId"])

    lamb = _client("lambda")
    for name in ("rag-ops-guard-query", "rag-ops-guard-ingest"):
        if _lambda_exists(lamb, name):
            lamb.delete_function(FunctionName=name)

    print("reset complete")


if __name__ == "__main__":
    main()
