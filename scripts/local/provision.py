from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import ClientError

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore
from rag_ops_guard.configstore.runtime import snapshot_for_values

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
DOC_BUCKET = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
LAMBDA_CODE_BUCKET = os.environ.get(
    "S3_LAMBDA_CODE_BUCKET", "rag-ops-guard-lambda-code-local"
)
VECTOR_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "rag-ops-guard-vectors-local")
VECTOR_INDEX = os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-v1")
CONFIG_TABLE = os.environ.get("CONFIG_TABLE", "rag-ops-config")
LAMBDA_ZIP_PATH = Path(
    os.environ.get("LAMBDA_ZIP_PATH", ".local/lambda-package.zip")
).resolve()
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-2b-unsloth-ud-q4-k-xl")
LAMBDA_PYTHON_VERSION = os.environ.get("LAMBDA_PYTHON_VERSION", "3.12")
LAMBDA_TIMEOUT_SECONDS = int(os.environ.get("LAMBDA_TIMEOUT_SECONDS", "180"))
LAMBDA_LLM_BASE_URL = os.environ.get("LAMBDA_LLM_BASE_URL", "http://llama-gen:8080/v1")
LAMBDA_EMBEDDING_BASE_URL = os.environ.get(
    "LAMBDA_EMBEDDING_BASE_URL", "http://llama-embed:8081/v1"
)
LAMBDA_EMBEDDING_MODEL = os.environ.get(
    "LAMBDA_EMBEDDING_MODEL",
    os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
)
LAMBDA_RERANKER_BASE_URL = os.environ.get(
    "LAMBDA_RERANKER_BASE_URL", "http://llama-rerank:8082"
)
LAMBDA_RERANKER_MODEL = os.environ.get(
    "LAMBDA_RERANKER_MODEL",
    os.environ.get("RERANKER_MODEL", "qwen3-reranker-0.6b"),
)
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


def s3_client() -> Any:
    return client("s3", config=Config(s3={"addressing_style": "path"}))


def ensure_bucket(bucket: str) -> None:
    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)


def ensure_s3() -> None:
    ensure_bucket(DOC_BUCKET)
    ensure_bucket(LAMBDA_CODE_BUCKET)


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
            dimension=int(os.environ.get("VECTOR_DIMENSION", "1024")),
            distanceMetric="cosine",
        )


def ensure_config_table() -> None:
    dynamodb = client("dynamodb")
    try:
        dynamodb.create_table(
            TableName=CONFIG_TABLE,
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
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceInUseException":
            raise


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


def _config_store(settings: Settings) -> DynamoDbConfigStore:
    return DynamoDbConfigStore(
        endpoint_url=ENDPOINT,
        region=REGION,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        table=CONFIG_TABLE,
    )


def active_config_hash() -> str:
    settings = Settings()
    store = _config_store(settings)
    head = store.get_head()
    if head is None:
        return snapshot_for_values(settings, {}, revision_no=None).config_hash
    values = store.get_revision_values(head.revision_no)
    return snapshot_for_values(settings, values, revision_no=head.revision_no).config_hash


def lambda_environment(config_hash: str | None = None) -> dict[str, str]:
    """Bootstrap only: behavioral tuning/floors are resolved from the config registry."""
    environment = {
        "APP_ENV": "local",
        "AWS_REGION": REGION,
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_ENDPOINT_URL": "http://floci:4566",
        "CONFIG_SOURCE": "db",
        "CONFIG_TABLE": CONFIG_TABLE,
        "CONFIG_HASH": config_hash or os.environ.get("CONFIG_HASH", "0" * 64),
        "CONFIG_HEAD_TTL_SECONDS": os.environ.get("CONFIG_HEAD_TTL_SECONDS", "45"),
        "CONFIG_MAX_STALE_SECONDS": os.environ.get("CONFIG_MAX_STALE_SECONDS", "300"),
        # Resource identity and process-topology endpoints remain bootstrap because host and
        # Lambda use different network addresses for the same physical services.
        "S3_DOCUMENT_BUCKET": DOC_BUCKET,
        "S3_VECTOR_BUCKET": VECTOR_BUCKET,
        "S3_VECTOR_INDEX": VECTOR_INDEX,
        "VECTOR_DIMENSION": os.environ.get("VECTOR_DIMENSION", "1024"),
        "LLM_BASE_URL": LAMBDA_LLM_BASE_URL,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_BASE_URL": LAMBDA_EMBEDDING_BASE_URL,
        "EMBEDDING_MODEL": LAMBDA_EMBEDDING_MODEL,
        "EMBEDDING_DIMENSION": os.environ.get("EMBEDDING_DIMENSION", "1024"),
        "RERANKER_BASE_URL": LAMBDA_RERANKER_BASE_URL,
        "RERANKER_MODEL": LAMBDA_RERANKER_MODEL,
    }
    environment.update(_langsmith_environment())
    return environment


def publish_lambda_code() -> str:
    payload = LAMBDA_ZIP_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    key = f"lambda/{digest}.zip"
    s3_client().put_object(
        Bucket=LAMBDA_CODE_BUCKET,
        Key=key,
        Body=payload,
        ContentType="application/zip",
        Metadata={"sha256": digest},
    )
    return key


def recreate_lambda(
    name: str,
    handler: str,
    role_arn: str,
    code_key: str,
    *,
    config_hash: str,
) -> str:
    lamb = client("lambda")
    with suppress(ClientError):
        lamb.delete_function(FunctionName=name)
    response = lamb.create_function(
        FunctionName=name,
        Runtime=f"python{LAMBDA_PYTHON_VERSION}",
        Role=role_arn,
        Handler=handler,
        Code={"S3Bucket": LAMBDA_CODE_BUCKET, "S3Key": code_key},
        Timeout=LAMBDA_TIMEOUT_SECONDS,
        MemorySize=1024,
        Environment={"Variables": lambda_environment(config_hash=config_hash)},
    )
    return str(response["FunctionArn"])


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
    response = client("lambda").invoke(
        FunctionName=function_name,
        Payload=b"{}",
    )
    raw = response["Payload"].read()
    function_error = response.get("FunctionError")
    if function_error:
        raise RuntimeError(
            f"Lambda {function_name} failed during direct invocation "
            f"({function_error}): {raw[:2000].decode('utf-8', errors='replace')}"
        )
    payload = parse_proxy_payload(raw)
    validate_invalid_request_payload(payload, source=f"Lambda {function_name}")


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


def recreate_api(query_arn: str, ingest_arn: str) -> str:
    api = client("apigatewayv2")
    lamb = client("lambda")
    for existing in api.get_apis().get("Items", []):
        if existing.get("Name") == "rag-ops-guard-local":
            api.delete_api(ApiId=existing["ApiId"])

    created = api.create_api(
        Name="rag-ops-guard-local",
        ProtocolType="HTTP",
    )
    api_id = str(created["ApiId"])
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
        with suppress(ClientError):
            lamb.add_permission(
                FunctionName=function_name,
                StatementId=statement,
                Action="lambda:InvokeFunction",
                Principal="apigateway.amazonaws.com",
            )
    api.create_stage(ApiId=api_id, StageName=DEFAULT_API_STAGE, AutoDeploy=True)

    endpoint = floci_execution_endpoint(ENDPOINT, api_id)
    probe_api_route(endpoint, "/v1/ingest")
    probe_api_route(endpoint, "/v1/query")
    Path(".local").mkdir(exist_ok=True)
    Path(".local/api-url").write_text(endpoint)
    return endpoint


def main() -> None:
    if not LAMBDA_ZIP_PATH.is_file():
        raise SystemExit("Lambda ZIP missing. Run make package-lambda first.")
    ensure_s3()
    ensure_vectors()
    ensure_config_table()
    config_hash = active_config_hash()
    code_key = publish_lambda_code()
    config_read_actions = [
        "dynamodb:GetItem",
        "dynamodb:BatchGetItem",
        "dynamodb:Query",
        "dynamodb:DescribeTable",
    ]
    ingest_role = ensure_role(
        "rag-ops-guard-ingest-role",
        [
            "s3:GetObject",
            "s3:PutObject",
            "s3vectors:PutVectors",
            "s3vectors:GetVectors",
            "s3vectors:DeleteVectors",
            *config_read_actions,
        ],
    )
    query_role = ensure_role(
        "rag-ops-guard-query-role",
        [
            "s3:GetObject",
            "s3vectors:QueryVectors",
            "s3vectors:GetVectors",
            *config_read_actions,
        ],
    )
    ingest_arn = recreate_lambda(
        "rag-ops-guard-ingest",
        "rag_ops_guard.handlers.ingest.handler",
        ingest_role,
        code_key,
        config_hash=config_hash,
    )
    query_arn = recreate_lambda(
        "rag-ops-guard-query",
        "rag_ops_guard.handlers.query.handler",
        query_role,
        code_key,
        config_hash=config_hash,
    )
    probe_lambda("rag-ops-guard-ingest")
    probe_lambda("rag-ops-guard-query")
    endpoint = recreate_api(query_arn, ingest_arn)
    print(f"Lambda package: s3://{LAMBDA_CODE_BUCKET}/{code_key}")
    print("Lambda direct invoke: ready")
    print(f"Lambda timeout: {LAMBDA_TIMEOUT_SECONDS}s")
    tracing = lambda_environment(config_hash=config_hash).get("LANGSMITH_TRACING") == "true"
    print(f"LangSmith tracing: {'enabled' if tracing else 'disabled'}")
    print(f"Config table: {CONFIG_TABLE}")
    print(f"Config hash: {config_hash}")
    print(f"Local API: {endpoint}")
    print("API data plane: ready")


if __name__ == "__main__":
    main()
