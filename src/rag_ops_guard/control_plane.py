from __future__ import annotations

import json
from typing import Any, Literal

import boto3
from pydantic import BaseModel, ConfigDict, Field, ValidationError

APPCONFIG_APPLICATION = "rag-ops-guard"
APPCONFIG_ENVIRONMENT = "runtime"
APPCONFIG_PROFILE = "control-plane"


class ControlPlaneResolutionError(RuntimeError):
    """The workload could not resolve a valid deployed platform control plane."""


class PlatformResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_table: str = Field(min_length=1)
    tenant_credential_table: str = Field(min_length=1)
    document_bucket: str = Field(min_length=1)
    vector_bucket: str = Field(min_length=1)
    vector_index_base: str = Field(min_length=1)


class ModelService(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)


class EmbeddingService(ModelService):
    dimension: int = Field(ge=1, le=4096)


class PlatformServices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: ModelService
    embedding: EmbeddingService
    reranker: ModelService


class BootstrapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_head_ttl_seconds: int = Field(ge=1, le=3600)
    config_max_stale_seconds: int = Field(ge=1, le=86400)
    fail_closed: bool = True


class ControlPlaneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    resources: PlatformResources
    services: PlatformServices
    secret_refs: dict[str, str] = Field(default_factory=dict)
    bootstrap: BootstrapPolicy


def _configuration_bytes(value: object) -> bytes:
    raw = value.read() if hasattr(value, "read") else value
    if isinstance(raw, str):
        return raw.encode("utf-8")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    raise ControlPlaneResolutionError("AppConfigData returned an unsupported configuration body")


def _named_resource_id(
    client: Any,
    operation: str,
    expected_name: str,
    **parameters: object,
) -> str:
    request = dict(parameters)
    matches: list[str] = []
    while True:
        response = getattr(client, operation)(**request)
        for item in response.get("Items", []):
            if item.get("Name") == expected_name and item.get("Id"):
                matches.append(str(item["Id"]))
        next_token = response.get("NextToken")
        if not next_token:
            break
        request["NextToken"] = str(next_token)

    if len(matches) != 1:
        raise ControlPlaneResolutionError(
            f"AppConfig resource {expected_name!r} resolved to {len(matches)} physical ids"
        )
    return matches[0]


def _resolve_physical_ids(appconfig: Any) -> tuple[str, str, str]:
    application_id = _named_resource_id(
        appconfig,
        "list_applications",
        APPCONFIG_APPLICATION,
    )
    environment_id = _named_resource_id(
        appconfig,
        "list_environments",
        APPCONFIG_ENVIRONMENT,
        ApplicationId=application_id,
    )
    profile_id = _named_resource_id(
        appconfig,
        "list_configuration_profiles",
        APPCONFIG_PROFILE,
        ApplicationId=application_id,
    )
    return application_id, environment_id, profile_id


def fetch_control_plane(
    *,
    management_client: Any | None = None,
    data_client: Any | None = None,
) -> ControlPlaneConfig:
    """Resolve the active platform control plane through the standard AWS SDK provider chain."""
    appconfig = management_client or boto3.client("appconfig")
    appconfigdata = data_client or boto3.client("appconfigdata")
    application_id, environment_id, profile_id = _resolve_physical_ids(appconfig)

    session = appconfigdata.start_configuration_session(
        ApplicationIdentifier=application_id,
        EnvironmentIdentifier=environment_id,
        ConfigurationProfileIdentifier=profile_id,
    )
    token = session.get("InitialConfigurationToken")
    if not token:
        raise ControlPlaneResolutionError("AppConfigData did not return an initial session token")

    response = appconfigdata.get_latest_configuration(ConfigurationToken=str(token))
    payload = _configuration_bytes(response.get("Configuration", b""))
    if not payload:
        raise ControlPlaneResolutionError("AppConfigData returned an empty initial configuration")

    try:
        decoded = json.loads(payload.decode("utf-8"))
        return ControlPlaneConfig.model_validate(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ControlPlaneResolutionError(
            "AppConfigData returned an invalid control-plane payload"
        ) from exc
