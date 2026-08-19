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


def fetch_control_plane(*, client: Any | None = None) -> ControlPlaneConfig:
    """Resolve the active platform control plane through the standard AWS SDK provider chain."""
    appconfigdata = client or boto3.client("appconfigdata")
    session = appconfigdata.start_configuration_session(
        ApplicationIdentifier=APPCONFIG_APPLICATION,
        EnvironmentIdentifier=APPCONFIG_ENVIRONMENT,
        ConfigurationProfileIdentifier=APPCONFIG_PROFILE,
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
