from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
DOC_BUCKET = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
VECTOR_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "rag-ops-guard-vectors-local")
VECTOR_INDEX = os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-v1")
LAMBDA_CODE_PATH = Path(os.environ.get("LAMBDA_CODE_PATH", ".local/lambda-package")).resolve()


def client(service: str, **kwargs: object) -> Any:
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **kwargs,
    )


def ensure_s3() -> None:
    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    try:
        s3.head_bucket(Bucket=DOC_BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=DOC_BUCKET)


def ensure_vectors() -> None:
    vectors = client("s3vectors")
    try:
        vectors.get_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    except ClientError:
        vectors.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    try:
        vectors.get_index(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX)
    except ClientError:
        vectors.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
        )


def ensure_role(name: str, actions: list[str]) -> str:
    iam = client("iam")
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role = iam.get_role(RoleName=name)["Role"]
    except ClientError:
        role = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust),
        )["Role"]
    iam.put_role_policy(
        RoleName=name,
        PolicyName=f"{name}-policy",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": actions,
                        "Resource": "*",
                    }
                ],
            }
        ),
    )
    return str(role["Arn"])


def lambda_environment() -> dict[str, str]:
    return {
        "APP_ENV": "local",
        "AWS_REGION": REGION,
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_ENDPOINT_URL": "http://floci:4566",
        "S3_DOCUMENT_BUCKET": DOC_BUCKET,
        "S3_VECTOR_BUCKET": VECTOR_BUCKET,
        "S3_VECTOR_INDEX": VECTOR_INDEX,
        "VECTOR_DIMENSION": "1024",
        "RETRIEVAL_TOP_K": "8",
        "RETRIEVAL_CONTEXT_K": "5",
        "CHUNK_TOKENS": "400",
        "CHUNK_OVERLAP": "60",
        "LLM_BASE_URL": "http://llama-gen:8080/v1",
        "LLM_MODEL": "qwen3-4b-rag",
        "LLM_MAX_TOKENS": "256",
        "LLM_TEMPERATURE": "0.7",
        "EMBEDDING_BASE_URL": "http://llama-embed:8081/v1",
        "EMBEDDING_MODEL": "qwen3-embedding-0.6b",
        "EMBEDDING_DIMENSION": "1024",
        "LANGSMITH_TRACING": "false",
    }


def recreate_lambda(name: str, handler: str, role_arn: str) -> str:
    lamb = client("lambda")
    try:
        lamb.delete_function(FunctionName=name)
    except ClientError:
        pass
    response = lamb.create_function(
        FunctionName=name,
        Runtime="python3.12",
        Role=role_arn,
        Handler=handler,
        Code={"S3Bucket": "hot-reload", "S3Key": str(LAMBDA_CODE_PATH)},
        Timeout=120,
        MemorySize=1024,
        Environment={"Variables": lambda_environment()},
    )
    return str(response["FunctionArn"])


def recreate_api(query_arn: str, ingest_arn: str) -> str:
    api = client("apigatewayv2")
    lamb = client("lambda")
    for existing in api.get_apis().get("Items", []):
        if existing.get("Name") == "rag-ops-guard-local":
            api.delete_api(ApiId=existing["ApiId"])

    created = api.create_api(Name="rag-ops-guard-local", ProtocolType="HTTP")
    api_id = created["ApiId"]
    for route_key, function_arn, statement in (
        ("POST /v1/query", query_arn, "AllowApiQuery"),
        ("POST /v1/ingest", ingest_arn, "AllowApiIngest"),
    ):
        integration = api.create_integration(
            ApiId=api_id,
            IntegrationType="AWS_PROXY",
            IntegrationUri=function_arn,
            PayloadFormatVersion="2.0",
        )
        api.create_route(
            ApiId=api_id,
            RouteKey=route_key,
            Target=f"integrations/{integration['IntegrationId']}",
        )
        function_name = function_arn.rsplit(":", 1)[-1]
        try:
            lamb.add_permission(
                FunctionName=function_name,
                StatementId=statement,
                Action="lambda:InvokeFunction",
                Principal="apigateway.amazonaws.com",
            )
        except ClientError:
            pass
    api.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)
    endpoint = str(created.get("ApiEndpoint") or "http://localhost:4566")
    Path(".local").mkdir(exist_ok=True)
    Path(".local/api-url").write_text(endpoint)
    return endpoint


def main() -> None:
    if not LAMBDA_CODE_PATH.exists():
        raise SystemExit("Lambda package missing. Run make package-lambda first.")
    ensure_s3()
    ensure_vectors()
    ingest_role = ensure_role(
        "rag-ops-guard-ingest-role",
        [
            "s3:GetObject",
            "s3:PutObject",
            "s3vectors:PutVectors",
            "s3vectors:GetVectors",
            "s3vectors:DeleteVectors",
        ],
    )
    query_role = ensure_role(
        "rag-ops-guard-query-role",
        [
            "s3:GetObject",
            "s3vectors:QueryVectors",
            "s3vectors:GetVectors",
        ],
    )
    ingest_arn = recreate_lambda(
        "rag-ops-guard-ingest",
        "rag_ops_guard.handlers.ingest.handler",
        ingest_role,
    )
    query_arn = recreate_lambda(
        "rag-ops-guard-query",
        "rag_ops_guard.handlers.query.handler",
        query_role,
    )
    endpoint = recreate_api(query_arn, ingest_arn)
    print(f"Local API: {endpoint}")


if __name__ == "__main__":
    main()
