from __future__ import annotations

import json
import os

import boto3
import pytest

from rag_ops_guard.control_plane import (
    APPCONFIG_APPLICATION,
    APPCONFIG_ENVIRONMENT,
    APPCONFIG_PROFILE,
    ControlPlaneConfig,
)

pytestmark = pytest.mark.integration


def test_floci_serves_canonical_appconfig_control_plane() -> None:
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
    client = boto3.client(
        "appconfigdata",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )

    session = client.start_configuration_session(
        ApplicationIdentifier=APPCONFIG_APPLICATION,
        EnvironmentIdentifier=APPCONFIG_ENVIRONMENT,
        ConfigurationProfileIdentifier=APPCONFIG_PROFILE,
    )
    token = str(session["InitialConfigurationToken"])
    response = client.get_latest_configuration(ConfigurationToken=token)
    payload = response["Configuration"].read()

    assert payload
    resolved = ControlPlaneConfig.model_validate(json.loads(payload))
    assert resolved.schema_version == 1
    assert resolved.resources.config_table == "rag-ops-config"
    assert resolved.resources.tenant_credential_table == "rag-ops-tenants"
    assert resolved.services.embedding.dimension == 1024
    assert resolved.bootstrap.fail_closed is True
