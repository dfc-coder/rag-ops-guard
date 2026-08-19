from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher

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
    """Materialize the S3 Vectors resources Floci 1.6.0 cannot create from CFN yet."""

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


def bootstrap_local_tenant(outputs: dict[str, str]) -> None:
    """Persist only an Argon2id hash for the locally supplied Phase 4 API key."""

    api_key = os.environ.get("RAG_OPS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "RAG_OPS_API_KEY is required for Phase 4 local provisioning; use key_id.secret format"
        )
    tenant_id = os.environ.get("RAG_OPS_TENANT_ID", "default").strip() or "default"
    key_id, separator, secret = api_key.partition(".")
    if not separator or not key_id or not secret:
        raise RuntimeError("RAG_OPS_API_KEY must use key_id.secret format")

    client("dynamodb").put_item(
        TableName=required_output(outputs, "TenantTableName"),
        Item={
            "key_id": {"S": key_id},
            "tenant_id": {"S": tenant_id},
            "token_hash": {"S": TenantTokenHasher().hash_token(secret)},
            "enabled": {"BOOL": True},
        },
    )
    print(f"Tenant credential: ready (tenant={tenant_id}, key_id={key_id})")


def _langsmith_environment() -> dict[str, str]:
    tracing_requested = os.environ.get("LANGSMITH_TRACING", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    values = {
        "LANGSMITH_TRACING": "true" if tracing_requested and api_key else "false",
        "LANGSMITH_PROJECT": os.environ.get("LANGSMITH_PROJECT", "rag-ops-guard-local"),
        "LANGSMITH_ENDPOINT": os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ),
    }
    if api_key:
        values["LANGSMITH_API_KEY"] = api_key
    workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID", "").strip()
    if workspace_id:
        values["LANGSMITH_WORKSPACE_ID"] = workspace_id
    return values


def apply_local_runtime_secrets(function_names: tuple[str, ...]) -> None:
    """Inject local-only observability secrets without serializing them into CFN."""

    lamb = client("lambda")
    desired = _langsmith_environment()
    for function_name in function_names:
        current = lamb.get_function_configuration(FunctionName=function_name)
        variables = dict(current.get("Environment", {}).get("Variables", {}))
        variables.pop("LANGSMITH_API_KEY", None)
        variables.pop("LANGSMITH_WORKSPACE_ID", None)
        variables.update(desired)
        lamb.update_function_configuration(
            FunctionName=function_name,
            Environment={"Variables": variables},
        )


def floci_execution_endpoint(base_endpoint: str, api_id: str) -> str:
    return f"{base_endpoint.rstrip('/')}/execute-api/{api_id}/{DEFAULT_API_STAGE}"


def main() -> None:
    outputs = stack_outputs()
    materialize_floci_s3_vectors(outputs)
    bootstrap_local_tenant(outputs)

    ingest_name = required_output(outputs, "IngestFunctionName")
    query_name = required_output(outputs, "QueryFunctionName")
    api_id = required_output(outputs, "ApiId")

    apply_local_runtime_secrets((ingest_name, query_name))
    endpoint = floci_execution_endpoint(ENDPOINT, api_id)

    Path(".local").mkdir(exist_ok=True)
    Path(".local/api-url").write_text(endpoint, encoding="utf-8")

    print(f"CDK stack: {STACK_NAME}")
    print("Floci S3 Vectors bridge: ready")
    print("Local runtime secrets: synchronized")
    print(f"Local API: {endpoint}")


if __name__ == "__main__":
    main()
