from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from rag_ops_guard.app import ingestion_service
from rag_ops_guard.configstore.runtime import EffectiveConfig
from rag_ops_guard.configstore.tenant_runtime import resolve_tenant_effective_config
from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import IngestRequest
from rag_ops_guard.observability.runtime import (
    INGEST_LOGGER,
    INGEST_METRICS,
    add_count,
    add_milliseconds,
    add_seconds,
    config_observability_fields,
    set_config_dimensions,
)
from rag_ops_guard.tenancy import ApiKeyAuthenticationError
from rag_ops_guard.tenancy.runtime_auth import MissingApiKeyError, request_context_from_event


def _internal_error_body(exc: Exception) -> str:
    del exc
    return json.dumps({"error": "internal_error"})


def _record_config_observability(
    effective: EffectiveConfig, resolve_latency_ms: float
) -> dict[str, object]:
    fields = config_observability_fields(
        revision_no=effective.revision_no,
        config_hash=effective.config_hash,
        source=effective.source,
    )
    set_config_dimensions(
        INGEST_METRICS,
        revision_no=effective.revision_no,
        config_hash=effective.config_hash,
        source=effective.source,
    )
    add_milliseconds(INGEST_METRICS, "ConfigResolveLatencyMs", resolve_latency_ms)
    if effective.source == "cache":
        add_count(INGEST_METRICS, "ConfigCacheHit")
    elif effective.source != "env":
        add_count(INGEST_METRICS, "ConfigCacheMiss")
    if effective.db_unavailable:
        add_count(INGEST_METRICS, "ConfigDbUnavailable")
    if effective.stale:
        add_count(INGEST_METRICS, "ConfigStaleServed")
    if effective.revision_age_s is not None:
        add_seconds(INGEST_METRICS, "ConfigRevisionAge", effective.revision_age_s)
    return dict(fields)


def _json_proxy(status_code: int, payload: dict[str, object]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    del context
    started = perf_counter()
    config_fields: dict[str, object] = {
        "config_revision": None,
        "config_hash": "unknown",
        "config_source": "unresolved",
    }
    set_config_dimensions(
        INGEST_METRICS,
        revision_no=None,
        config_hash="unknown",
        source="unresolved",
    )
    try:
        request_context = request_context_from_event(event)
        resolve_started = perf_counter()
        effective = resolve_tenant_effective_config(request_context.tenant_id)
        config_fields = _record_config_observability(
            effective,
            (perf_counter() - resolve_started) * 1000,
        )
        config_fields["tenant_id"] = request_context.tenant_id
        add_count(INGEST_METRICS, "IngestRequestCount")

        body = event.get("body", event)
        payload = json.loads(body) if isinstance(body, str) else body
        request = IngestRequest.model_validate(payload)
        response = ingestion_service(request_context).ingest(request.s3_key)
        add_count(INGEST_METRICS, "IngestedCount" if response.status == "ingested" else "NoOpCount")
        INGEST_LOGGER.info(
            "ingest_completed",
            extra={
                **config_fields,
                "status": response.status,
                "logical_id": response.logical_id,
                "version": response.version,
                "chunks": response.chunks,
            },
        )
        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": response.model_dump_json(),
        }
    except (MissingApiKeyError, ApiKeyAuthenticationError):
        add_count(INGEST_METRICS, "UnauthorizedCount")
        INGEST_LOGGER.warning("unauthorized_ingest_request", extra=config_fields)
        return _json_proxy(401, {"error": "unauthorized"})
    except DocumentValidationError as exc:
        add_count(INGEST_METRICS, "InvalidDocumentCount")
        INGEST_LOGGER.warning(
            "invalid_document",
            extra={**config_fields, "error_type": type(exc).__name__},
        )
        return {
            "statusCode": 422,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_document", "detail": str(exc)}),
        }
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        add_count(INGEST_METRICS, "InvalidRequestCount")
        INGEST_LOGGER.warning(
            "invalid_ingest_request",
            extra={**config_fields, "error_type": type(exc).__name__},
        )
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_request", "detail": str(exc)}),
        }
    except Exception as exc:
        add_count(INGEST_METRICS, "InternalErrorCount")
        INGEST_LOGGER.exception(
            "ingest_failed",
            extra={**config_fields, "error_type": type(exc).__name__},
        )
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": _internal_error_body(exc),
        }
    finally:
        add_milliseconds(INGEST_METRICS, "IngestLatencyMs", (perf_counter() - started) * 1000)
        INGEST_METRICS.flush_metrics()
