from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import boto3
import httpx
from botocore.config import Config

from rag_ops_guard.app import embeddings, reranker, vector_store
from rag_ops_guard.configstore.runtime import EffectiveConfig
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore
from rag_ops_guard.runtime_settings import runtime_control_plane
from rag_ops_guard.tenancy import KeyLayout, RequestContext


def _check(operation: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return {"ok": True, **operation()}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _aws_client(service: str, **kwargs: object) -> Any:
    client_factory = cast(Any, boto3.client)
    return client_factory(service, **kwargs)


def _endpoint(client: Any) -> str:
    return str(getattr(getattr(client, "meta", None), "endpoint_url", ""))


def _probe_dynamodb(effective: EffectiveConfig, context: RequestContext) -> dict[str, object]:
    control_plane = runtime_control_plane()
    table = control_plane.resources.config_table
    client = _aws_client("dynamodb")
    client.describe_table(TableName=table)
    store = TenantDynamoDbConfigStore(table=table, tenant_id=context.tenant_id)
    head = store.get_head()
    if head is None:
        raise RuntimeError(f"tenant {context.tenant_id} config table has no HEAD revision")
    if effective.revision_no is not None and head.revision_no != effective.revision_no:
        raise RuntimeError(
            f"live HEAD revision {head.revision_no} != effective revision {effective.revision_no}"
        )
    return {
        "endpoint": _endpoint(client),
        "table": table,
        "tenant_id": context.tenant_id,
        "head_revision": head.revision_no,
    }


def _probe_s3(effective: EffectiveConfig) -> dict[str, object]:
    settings = effective.settings
    client = _aws_client("s3", config=Config(s3={"addressing_style": "path"}))
    response = client.list_objects_v2(Bucket=settings.s3_document_bucket, MaxKeys=1)
    return {
        "endpoint": _endpoint(client),
        "bucket": settings.s3_document_bucket,
        "sample_count": len(response.get("Contents", [])),
    }


def _probe_s3vectors(
    effective: EffectiveConfig,
    vector: list[float],
    context: RequestContext,
) -> dict[str, object]:
    settings = effective.settings
    index_name = KeyLayout(context.tenant_id).vector_index(settings.s3_vector_index)
    client = _aws_client("s3vectors")
    client.get_index(
        vectorBucketName=settings.s3_vector_bucket,
        indexName=index_name,
    )
    hits = vector_store(context).query(vector, top_k=1)
    return {
        "endpoint": _endpoint(client),
        "bucket": settings.s3_vector_bucket,
        "index": index_name,
        "query_hits": len(hits),
    }


def _probe_llm(effective: EffectiveConfig) -> dict[str, object]:
    settings = effective.settings
    response = httpx.get(f"{settings.llm_base_url.rstrip('/')}/models", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    ids = {
        str(item.get("id"))
        for item in models or []
        if isinstance(item, dict) and item.get("id")
    }
    if settings.llm_model not in ids:
        raise RuntimeError(
            "LLM identity mismatch: "
            f"expected {settings.llm_model!r}, server reports {sorted(ids)!r}"
        )
    return {
        "endpoint": settings.llm_base_url,
        "model": settings.llm_model,
    }


def _probe_embeddings(
    effective: EffectiveConfig,
    context: RequestContext,
) -> tuple[dict[str, object], list[float]]:
    settings = effective.settings
    vector = embeddings(context).embed_query("rag ops guard physical connectivity probe")
    if len(vector) != settings.embedding_dimension:
        raise RuntimeError(
            f"embedding dimension mismatch: {len(vector)} != {settings.embedding_dimension}"
        )
    return (
        {
            "endpoint": settings.embedding_base_url,
            "model": settings.embedding_model,
            "dimension": len(vector),
        },
        vector,
    )


def _probe_reranker(effective: EffectiveConfig, context: RequestContext) -> dict[str, object]:
    settings = effective.settings
    grades = reranker(context).grade(
        "rag ops guard physical connectivity probe",
        ["rag ops guard physical connectivity probe"],
    )
    if len(grades) != 1:
        raise RuntimeError(f"reranker returned {len(grades)} grades, expected 1")
    return {
        "endpoint": settings.reranker_base_url,
        "model": settings.reranker_model,
        "score": grades[0].score,
    }


def probe_runtime_connectivity(
    effective: EffectiveConfig,
    context: RequestContext,
) -> dict[str, object]:
    """Exercise every physical dependency from one authenticated tenant runtime."""
    checks: dict[str, dict[str, object]] = {}
    checks["dynamodb"] = _check(lambda: _probe_dynamodb(effective, context))
    checks["s3"] = _check(lambda: _probe_s3(effective))
    checks["llm"] = _check(lambda: _probe_llm(effective))

    vector: list[float] | None = None
    try:
        embedding_detail, vector = _probe_embeddings(effective, context)
        checks["embeddings"] = {"ok": True, **embedding_detail}
    except Exception as exc:
        checks["embeddings"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    checks["reranker"] = _check(lambda: _probe_reranker(effective, context))
    if vector is None:
        checks["s3vectors"] = {
            "ok": False,
            "error_type": "DependencyUnavailable",
            "error": "embedding probe failed; vector query was not attempted",
        }
    else:
        checks["s3vectors"] = _check(lambda: _probe_s3vectors(effective, vector, context))

    return {
        "ok": all(bool(check.get("ok")) for check in checks.values()),
        "tenant_id": context.tenant_id,
        "config_revision": effective.revision_no,
        "config_hash": effective.config_hash,
        "config_source": effective.source,
        "checks": checks,
    }
