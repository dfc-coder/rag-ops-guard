from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
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


def _one_named(items: list[dict[str, Any]], name: str, resource: str) -> str | None:
    matches = [str(item["Id"]) for item in items if item.get("Name") == name and item.get("Id")]
    if len(matches) > 1:
        raise RuntimeError(f"Floci has multiple {resource} resources named {name!r}")
    return matches[0] if matches else None


def materialize_floci_appconfig(outputs: dict[str, str]) -> None:
    """Materialize CDK-declared AppConfig resources Floci 1.6.0 cannot create from CFN yet."""

    application_name = required_output(outputs, "AppConfigApplicationName")
    environment_name = required_output(outputs, "AppConfigEnvironmentName")
    profile_name = required_output(outputs, "AppConfigProfileName")
    payload = required_output(outputs, "AppConfigControlPlanePayload")
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CDK AppConfigControlPlanePayload output is invalid JSON") from exc

    appconfig = client("appconfig")
    appconfigdata = client("appconfigdata")

    application_id = _one_named(
        list(appconfig.list_applications().get("Items", [])),
        application_name,
        "AppConfig application",
    )
    if application_id is None:
        application_id = str(
            appconfig.create_application(
                Name=application_name,
                Description="RAG Ops Guard platform discovery control plane",
            )["Id"]
        )

    environment_id = _one_named(
        list(appconfig.list_environments(ApplicationId=application_id).get("Items", [])),
        environment_name,
        "AppConfig environment",
    )
    if environment_id is None:
        environment_id = str(
            appconfig.create_environment(
                ApplicationId=application_id,
                Name=environment_name,
                Description="Runtime workload control plane",
            )["Id"]
        )

    profile_id = _one_named(
        list(appconfig.list_configuration_profiles(ApplicationId=application_id).get("Items", [])),
        profile_name,
        "AppConfig profile",
    )
    if profile_id is None:
        profile_id = str(
            appconfig.create_configuration_profile(
                ApplicationId=application_id,
                Name=profile_name,
                LocationUri="hosted",
                Type="AWS.Freeform",
                Description="Non-secret platform discovery configuration",
            )["Id"]
        )

    current_payload = b""
    try:
        session = appconfigdata.start_configuration_session(
            ApplicationIdentifier=application_id,
            EnvironmentIdentifier=environment_id,
            ConfigurationProfileIdentifier=profile_id,
        )
        current = appconfigdata.get_latest_configuration(
            ConfigurationToken=str(session["InitialConfigurationToken"])
        )
        current_payload = current["Configuration"].read()
    except (ClientError, KeyError):
        current_payload = b""

    desired_payload = payload.encode("utf-8")
    if current_payload == desired_payload:
        return

    version = appconfig.create_hosted_configuration_version(
        ApplicationId=application_id,
        ConfigurationProfileId=profile_id,
        Content=desired_payload,
        ContentType="application/json",
        Description="Schema v1 RAG Ops Guard platform discovery payload",
    )
    appconfig.start_deployment(
        ApplicationId=application_id,
        EnvironmentId=environment_id,
        ConfigurationProfileId=profile_id,
        ConfigurationVersion=str(version["VersionNumber"]),
        DeploymentStrategyId="AppConfig.AllAtOnce",
        Description="Deploy the active RAG Ops Guard runtime control plane",
    )


def materialize_floci_s3_vectors(outputs: dict[str, str]) -> None:
    """Materialize CDK-declared resources Floci 1.6.0 cannot create from CFN yet.

    The function name is retained as the CI compatibility entrypoint from Phase 4. It now
    materializes both S3 Vectors and AppConfig from canonical CDK stack outputs.
    """

    bucket = required_output(outputs, "VectorBucketName")
    raw_indexes = required_output(outputs, "TenantVectorIndexNames")
    dimension = int(required_output(outputs, "VectorDimension"))
    indexes = json.loads(raw_indexes)
    if not isinstance(indexes, list) or not indexes or not all(isinstance(item, str) for item in indexes):
        raise RuntimeError("CDK TenantVectorIndexNames output is invalid")
    vectors = client("s3vectors")

    try:
        vectors.get_vector_bucket(vectorBucketName=bucket)
    except ClientError:
        vectors.create_vector_bucket(vectorBucketName=bucket)

    for index in indexes:
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

    materialize_floci_appconfig(outputs)


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

    ingest_name = required_output(outputs, "IngestFunctionName")
    query_name = required_output(outputs, "QueryFunctionName")
    api_id = required_output(outputs, "ApiId")

    apply_local_runtime_secrets((ingest_name, query_name))
    endpoint = floci_execution_endpoint(ENDPOINT, api_id)

    Path(".local").mkdir(exist_ok=True)
    Path(".local/api-url").write_text(endpoint, encoding="utf-8")

    print(f"CDK stack: {STACK_NAME}")
    print("Floci CDK compatibility bridges: S3 Vectors + AppConfig ready")
    print("Tenant credentials: managed separately by tenant-create/tenant-sync")
    print("Local runtime secrets: synchronized")
    print(f"Local API: {endpoint}")


if __name__ == "__main__":
    main()
