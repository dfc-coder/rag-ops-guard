from __future__ import annotations

import os
from typing import Any, Callable

import boto3
import httpx
from botocore.config import Config

from rag_ops_guard.app import embeddings, reranker, vector_store
from rag_ops_guard.configstore.dynamo_store import DynamoDbConfigStore
from rag_ops_guard.configstore.runtime import EffectiveConfig


def _check(operation: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return {"ok": True, **operation()}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _aws_client(service: str, effective: EffectiveConfig, **kwargs: object) -> Any:
    settings = effective.settings
    return boto3.client(
        service,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
        **kwargs,
    )


def _probe_dynamodb(effective: EffectiveConfig) -> dict[str, object]:
    settings = effective.settings
    table = os.environ.get("CONFIG_TABLE", "rag-ops-config")
    client = _aws_client("dynamodb", effective)
    client.describe_table(TableName=table)
    store = DynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=table,
    )
    head = store.get_head()
    if head is None:
        raise RuntimeError("config table has no HEAD revision")
    if effective.revision_no is not None and head.revision_no != effective.revision_no:
        raise RuntimeError(
            f"live HEAD revision {head.revision_no} != effective revision {effective.revision_no}"
        )
    return {
        "endpoint": settings.aws_endpoint_url,
        "table": table,
        "head_revision": head.revision_no,
    }


def _probe_s3(effective: EffectiveConfig) -> dict[str, object]:
    settings = effective.settings
    client = _aws_client(
        "s3",
        effective,
        config=Config(s3={"addressing_style": "path"}),
    )
    response = client.list_objects_v2(Bucket=settings.s3_document_bucket, MaxKeys=1)
    return {
        "endpoint": settings.aws_endpoint_url,
        "bucket": settings.s3_document_bucket,
        "sample_count": len(response.get("Contents", [])),
    }


def _probe_s3vectors(effective: EffectiveConfig, vector: list[float]) -> dict[str, object]:
    settings = effective.settings
    client = _aws_client("s3vectors", effective)
    client.get_index(
        vectorBucketName=settings.s3_vector_bucket,
        indexName=settings.s3_vector_index,
    )
    hits = vector_store().query(vector, top_k=1)
    return {
        "endpoint": settings.aws_endpoint_url,
        "bucket": settings.s3_vector_bucket,
        "index": settings.s3_vector_index,
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
            f"LLM identity mismatch: expected {settings.llm_model!r}, server reports {sorted(ids)!r}"
        )
    return {
        "endpoint": settings.llm_base_url,
        "model": settings.llm_model,
    }


def _probe_embeddings(effective: EffectiveConfig) -> tuple[dict[str, object], list[float]]:
    settings = effective.settings
    vector = embeddings().embed_query("rag ops guard physical connectivity probe")
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


def _probe_reranker(effective: EffectiveConfig) -> dict[str, object]:
    settings = effective.settings
    grades = reranker().grade(
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


def probe_runtime_connectivity(effective: EffectiveConfig) -> dict[str, object]:
    """Exercise every physical dependency from the Lambda runtime network namespace."""
    checks: dict[str, dict[str, object]] = {}
    checks["dynamodb"] = _check(lambda: _probe_dynamodb(effective))
    checks["s3"] = _check(lambda: _probe_s3(effective))
    checks["llm"] = _check(lambda: _probe_llm(effective))

    vector: list[float] | None = None
    try:
        embedding_detail, vector = _probe_embeddings(effective)
        checks["embeddings"] = {"ok": True, **embedding_detail}
    except Exception as exc:
        checks["embeddings"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    checks["reranker"] = _check(lambda: _probe_reranker(effective))
    if vector is None:
        checks["s3vectors"] = {
            "ok": False,
            "error_type": "DependencyUnavailable",
            "error": "embedding probe failed; vector query was not attempted",
        }
    else:
        checks["s3vectors"] = _check(lambda: _probe_s3vectors(effective, vector))

    return {
        "ok": all(bool(check.get("ok")) for check in checks.values()),
        "config_revision": effective.revision_no,
        "config_hash": effective.config_hash,
        "config_source": effective.source,
        "checks": checks,
    }
