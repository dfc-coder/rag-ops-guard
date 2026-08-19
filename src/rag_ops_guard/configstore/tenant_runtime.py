from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import cast

import boto3
from botocore.config import Config

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.runtime import (
    DEFAULT_HEAD_TTL_SECONDS,
    DEFAULT_MAX_STALE_SECONDS,
    ConfigSnapshot,
    ConfigSource,
    EffectiveConfig,
    RuntimeConfigResolver,
)
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.tenancy import KeyLayout

SnapshotLoader = Callable[[], ConfigSnapshot | None]


def settings_from_process_environment() -> Settings:
    """Phase-6.1 compatibility adapter; removed by Phase 6.2."""
    values: dict[str, object] = {}
    for field_name in Settings.model_fields:
        environment_name = field_name.upper()
        if environment_name in os.environ:
            values[field_name] = os.environ[environment_name]
    return Settings.model_validate(values)


def tenant_snapshot_key(tenant_id: str) -> str:
    return f"{KeyLayout(tenant_id).tenant_prefix}config/runtime-snapshot.json"


def tenant_snapshot_path(tenant_id: str) -> Path:
    return Path(".local") / f"config-snapshot-{KeyLayout(tenant_id).tenant_id}.json"


def write_tenant_snapshot(snapshot: ConfigSnapshot, tenant_id: str) -> Path:
    target = tenant_snapshot_path(tenant_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(snapshot.to_json(), encoding="utf-8")
    return target


def upload_tenant_snapshot(snapshot: ConfigSnapshot, bootstrap: Settings, tenant_id: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=bootstrap.aws_endpoint_url or None,
        region_name=bootstrap.aws_region,
        aws_access_key_id=bootstrap.aws_access_key_id,
        aws_secret_access_key=bootstrap.aws_secret_access_key.get_secret_value(),
        config=Config(s3={"addressing_style": "path"}),
    )
    client.put_object(
        Bucket=bootstrap.s3_document_bucket,
        Key=tenant_snapshot_key(tenant_id),
        Body=snapshot.to_json().encode("utf-8"),
        ContentType="application/json",
    )


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _s3_loader(bootstrap: Settings, tenant_id: str) -> SnapshotLoader:
    def load() -> ConfigSnapshot | None:
        client = boto3.client(
            "s3",
            endpoint_url=bootstrap.aws_endpoint_url or None,
            region_name=bootstrap.aws_region,
            aws_access_key_id=bootstrap.aws_access_key_id,
            aws_secret_access_key=bootstrap.aws_secret_access_key.get_secret_value(),
            config=Config(s3={"addressing_style": "path"}),
        )
        response = client.get_object(
            Bucket=bootstrap.s3_document_bucket,
            Key=tenant_snapshot_key(tenant_id),
        )
        return ConfigSnapshot.from_json(response["Body"].read().decode("utf-8"))

    return load


def _baked_loader(tenant_id: str) -> SnapshotLoader:
    def load() -> ConfigSnapshot | None:
        path = tenant_snapshot_path(tenant_id)
        if not path.is_file():
            return None
        return ConfigSnapshot.from_json(path.read_text(encoding="utf-8"))

    return load


@lru_cache(maxsize=32)
def _tenant_resolver(tenant_id: str) -> RuntimeConfigResolver:
    tenant = KeyLayout(tenant_id).tenant_id
    bootstrap = settings_from_process_environment()
    source_raw = os.environ.get("CONFIG_SOURCE", "db").strip().casefold()
    if source_raw not in {"db", "env"}:
        raise ValueError("CONFIG_SOURCE must be 'db' or 'env'")
    source = cast(ConfigSource, source_raw)
    store = TenantDynamoDbConfigStore(
        endpoint_url=bootstrap.aws_endpoint_url,
        region=bootstrap.aws_region,
        access_key=bootstrap.aws_access_key_id,
        secret_key=bootstrap.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
        tenant_id=tenant,
    )
    return RuntimeConfigResolver(
        store=store,
        bootstrap=bootstrap,
        source=source,
        head_ttl_s=_float_env("CONFIG_HEAD_TTL_SECONDS", DEFAULT_HEAD_TTL_SECONDS),
        max_stale_s=_float_env("CONFIG_MAX_STALE_SECONDS", DEFAULT_MAX_STALE_SECONDS),
        s3_loader=_s3_loader(bootstrap, tenant),
        baked_loader=_baked_loader(tenant),
    )


def resolve_tenant_effective_config(tenant_id: str) -> EffectiveConfig:
    return _tenant_resolver(KeyLayout(tenant_id).tenant_id).resolve()


def reset_tenant_effective_config_cache() -> None:
    _tenant_resolver.cache_clear()
