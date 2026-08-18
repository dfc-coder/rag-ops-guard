from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.dynamo_store import ConfigHead, DynamoDbConfigStore
from rag_ops_guard.configstore.hashing import content_hash
from rag_ops_guard.configstore.registry import registry_entries

ConfigSource = Literal["db", "env"]
ResolvedSource = Literal["dynamodb", "cache", "s3_snapshot", "baked", "code_default", "env"]
GLOBAL_SCOPE = "GLOBAL"
DEFAULT_HEAD_TTL_SECONDS = 45.0
DEFAULT_MAX_STALE_SECONDS = 300.0
SNAPSHOT_KEY = "config/runtime-snapshot.json"

# These values bind a process to its execution environment or external resources. They remain
# bootstrap inputs while Phase 2 moves behavioral/tuning policy to the versioned registry.
BOOTSTRAP_FIELDS = frozenset(
    {
        "app_env",
        "aws_region",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_endpoint_url",
        "s3_document_bucket",
        "s3_vector_bucket",
        "s3_vector_index",
        "vector_dimension",
        "vector_distance_metric",
        "llm_base_url",
        "llm_model",
        "embedding_base_url",
        "embedding_model",
        "embedding_dimension",
        "reranker_base_url",
        "reranker_model",
        "floci_host_port",
        "llama_gen_host_port",
        "llama_embed_host_port",
        "llama_rerank_host_port",
        "llama_ctx_size",
        "llama_parallel",
        "ragas_judge_provider",
        "ragas_judge_model",
        "ragas_judge_base_url",
        "ragas_judge_api_key",
        "ragas_judge_timeout_seconds",
        "ragas_operation_timeout_seconds",
        "ragas_max_retries",
        "ragas_max_wait_seconds",
        "ragas_max_workers",
        "ragas_suite_timeout_seconds",
        "langsmith_tracing",
        "langsmith_project",
        "langsmith_endpoint",
        "langsmith_api_key",
        "langsmith_workspace_id",
    }
)


class ConfigStore(Protocol):
    def get_head(self) -> ConfigHead | None: ...

    def get_revision_values(self, revision_no: int) -> dict[str, object]: ...


@dataclass(frozen=True)
class ConfigSnapshot:
    revision_no: int | None
    config_hash: str
    values: dict[str, object]
    created_at: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "revision_no": self.revision_no,
                "config_hash": self.config_hash,
                "values": self.values,
                "created_at": self.created_at,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> ConfigSnapshot:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("values"), dict):
            raise ValueError("invalid configuration snapshot")
        revision = payload.get("revision_no")
        return cls(
            revision_no=int(revision) if revision is not None else None,
            config_hash=str(payload.get("config_hash") or ""),
            values={str(key): value for key, value in payload["values"].items()},
            created_at=float(payload.get("created_at") or 0.0),
        )


@dataclass(frozen=True)
class EffectiveConfig:
    settings: Settings
    config_hash: str
    revision_no: int | None
    source: ResolvedSource
    stale: bool = False
    stale_age_s: float = 0.0
    fail_closed: bool = False


@dataclass(frozen=True)
class _CachedRevision:
    settings: Settings
    config_hash: str
    hydrated_at: float


def managed_field_names() -> tuple[str, ...]:
    entries = registry_entries()
    return tuple(
        sorted(
            name
            for name, entry in entries.items()
            if entry.sensitivity != "secret" and name not in BOOTSTRAP_FIELDS
        )
    )


def managed_config_values(settings: Settings) -> dict[str, object]:
    return {name: getattr(settings, name) for name in managed_field_names()}


def effective_config_hash(settings: Settings) -> str:
    return content_hash(managed_config_values(settings))


def _code_default_settings() -> Settings:
    return Settings.model_validate({})


def _merged_settings(bootstrap: Settings, managed_values: Mapping[str, object]) -> Settings:
    defaults = _code_default_settings()
    values: dict[str, object] = {
        name: getattr(defaults, name) for name in Settings.model_fields
    }
    for name in BOOTSTRAP_FIELDS:
        values[name] = getattr(bootstrap, name)
    allowed = set(managed_field_names())
    for name, value in managed_values.items():
        if name in allowed:
            values[name] = value
    return Settings.model_validate(values)


def snapshot_for_values(
    bootstrap: Settings,
    values: Mapping[str, object],
    *,
    revision_no: int | None,
    created_at: float | None = None,
) -> ConfigSnapshot:
    settings = _merged_settings(bootstrap, values)
    managed = managed_config_values(settings)
    return ConfigSnapshot(
        revision_no=revision_no,
        config_hash=content_hash(managed),
        values=managed,
        created_at=time.time() if created_at is None else created_at,
    )


def write_local_snapshot(snapshot: ConfigSnapshot, path: Path | None = None) -> Path:
    target = path or Path(".local/config-snapshot.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(snapshot.to_json(), encoding="utf-8")
    return target


def upload_s3_snapshot(snapshot: ConfigSnapshot, bootstrap: Settings) -> None:
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
        Key=SNAPSHOT_KEY,
        Body=snapshot.to_json().encode("utf-8"),
        ContentType="application/json",
    )


class RuntimeConfigResolver:
    def __init__(
        self,
        *,
        store: ConfigStore,
        bootstrap: Settings,
        source: ConfigSource = "db",
        head_ttl_s: float = DEFAULT_HEAD_TTL_SECONDS,
        max_stale_s: float = DEFAULT_MAX_STALE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        s3_loader: Callable[[], ConfigSnapshot | None] | None = None,
        baked_loader: Callable[[], ConfigSnapshot | None] | None = None,
    ) -> None:
        if head_ttl_s <= 0:
            raise ValueError("head_ttl_s must be > 0")
        if max_stale_s <= 0:
            raise ValueError("max_stale_s must be > 0")
        self._store = store
        self._bootstrap = bootstrap
        self._source = source
        self._head_ttl_s = head_ttl_s
        self._max_stale_s = max_stale_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._s3_loader = s3_loader or (lambda: None)
        self._baked_loader = baked_loader or (lambda: None)
        self._head: ConfigHead | None = None
        self._head_checked_at: float | None = None
        self._cache: dict[tuple[str, int], _CachedRevision] = {}
        self._last_good_key: tuple[str, int] | None = None

    def resolve(self) -> EffectiveConfig:
        now = self._clock()
        if self._source == "env":
            return EffectiveConfig(
                settings=self._bootstrap,
                config_hash=effective_config_hash(self._bootstrap),
                revision_no=None,
                source="env",
            )

        if (
            self._head is not None
            and self._head_checked_at is not None
            and now - self._head_checked_at < self._head_ttl_s
        ):
            cached = self._cache.get((GLOBAL_SCOPE, self._head.revision_no))
            if cached is not None:
                return self._from_cached(self._head.revision_no, cached, now, stale=False)

        try:
            head = self._store.get_head()
            self._head_checked_at = now
            self._head = head
            if head is not None:
                key = (GLOBAL_SCOPE, head.revision_no)
                cached = self._cache.get(key)
                if cached is not None:
                    self._last_good_key = key
                    return self._from_cached(head.revision_no, cached, now, stale=False)
                values = self._store.get_revision_values(head.revision_no)
                settings = _merged_settings(self._bootstrap, values)
                digest = effective_config_hash(settings)
                cached = _CachedRevision(settings=settings, config_hash=digest, hydrated_at=now)
                self._cache[key] = cached
                self._last_good_key = key
                return EffectiveConfig(
                    settings=settings,
                    config_hash=digest,
                    revision_no=head.revision_no,
                    source="dynamodb",
                )
        except (BotoCoreError, ClientError, OSError, RuntimeError, TimeoutError):
            pass

        if self._last_good_key is not None:
            cached = self._cache[self._last_good_key]
            return self._from_cached(
                self._last_good_key[1],
                cached,
                now,
                stale=True,
            )

        fallbacks: tuple[
            tuple[ResolvedSource, Callable[[], ConfigSnapshot | None]], ...
        ] = (
            ("s3_snapshot", self._s3_loader),
            ("baked", self._baked_loader),
        )
        for source, loader in fallbacks:
            try:
                snapshot = loader()
            except (BotoCoreError, ClientError, OSError, ValueError, json.JSONDecodeError):
                snapshot = None
            if snapshot is None:
                continue
            settings = _merged_settings(self._bootstrap, snapshot.values)
            digest = effective_config_hash(settings)
            if snapshot.config_hash and snapshot.config_hash != digest:
                continue
            stale_age = max(0.0, self._wall_clock() - snapshot.created_at)
            return EffectiveConfig(
                settings=settings,
                config_hash=digest,
                revision_no=snapshot.revision_no,
                source=source,
                stale=True,
                stale_age_s=stale_age,
                fail_closed=stale_age > self._max_stale_s,
            )

        defaults = _merged_settings(self._bootstrap, {})
        return EffectiveConfig(
            settings=defaults,
            config_hash=effective_config_hash(defaults),
            revision_no=None,
            source="code_default",
        )

    def _from_cached(
        self,
        revision_no: int,
        cached: _CachedRevision,
        now: float,
        *,
        stale: bool,
    ) -> EffectiveConfig:
        stale_age = max(0.0, now - cached.hydrated_at) if stale else 0.0
        return EffectiveConfig(
            settings=cached.settings,
            config_hash=cached.config_hash,
            revision_no=revision_no,
            source="cache",
            stale=stale,
            stale_age_s=stale_age,
            fail_closed=stale and stale_age > self._max_stale_s,
        )


def _load_snapshot_file(path: Path) -> ConfigSnapshot | None:
    if not path.is_file():
        return None
    return ConfigSnapshot.from_json(path.read_text(encoding="utf-8"))


def _default_baked_loader() -> ConfigSnapshot | None:
    configured = os.environ.get("CONFIG_BAKED_SNAPSHOT", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(".local/config-snapshot.json"),
        Path(__file__).with_name("baked_snapshot.json"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        snapshot = _load_snapshot_file(candidate)
        if snapshot is not None:
            return snapshot
    return None


def _default_s3_loader(bootstrap: Settings) -> Callable[[], ConfigSnapshot | None]:
    def load() -> ConfigSnapshot | None:
        client = boto3.client(
            "s3",
            endpoint_url=bootstrap.aws_endpoint_url or None,
            region_name=bootstrap.aws_region,
            aws_access_key_id=bootstrap.aws_access_key_id,
            aws_secret_access_key=bootstrap.aws_secret_access_key.get_secret_value(),
            config=Config(s3={"addressing_style": "path"}),
        )
        response = client.get_object(Bucket=bootstrap.s3_document_bucket, Key=SNAPSHOT_KEY)
        return ConfigSnapshot.from_json(response["Body"].read().decode("utf-8"))

    return load


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


@lru_cache(maxsize=1)
def _global_resolver() -> RuntimeConfigResolver:
    bootstrap = Settings()
    source_raw = os.environ.get("CONFIG_SOURCE", "db").strip().casefold()
    if source_raw not in {"db", "env"}:
        raise ValueError("CONFIG_SOURCE must be 'db' or 'env'")
    source = cast(ConfigSource, source_raw)
    store = DynamoDbConfigStore(
        endpoint_url=bootstrap.aws_endpoint_url,
        region=bootstrap.aws_region,
        access_key=bootstrap.aws_access_key_id,
        secret_key=bootstrap.aws_secret_access_key.get_secret_value(),
        table=os.environ.get("CONFIG_TABLE", "rag-ops-config"),
    )
    return RuntimeConfigResolver(
        store=store,
        bootstrap=bootstrap,
        source=source,
        head_ttl_s=_float_env("CONFIG_HEAD_TTL_SECONDS", DEFAULT_HEAD_TTL_SECONDS),
        max_stale_s=_float_env("CONFIG_MAX_STALE_SECONDS", DEFAULT_MAX_STALE_SECONDS),
        s3_loader=_default_s3_loader(bootstrap),
        baked_loader=_default_baked_loader,
    )


def resolve_effective_config() -> EffectiveConfig:
    return _global_resolver().resolve()


def reset_effective_config_cache() -> None:
    _global_resolver.cache_clear()
