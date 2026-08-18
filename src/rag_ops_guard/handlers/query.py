from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from langsmith import get_current_run_tree, traceable, tracing_context
from pydantic import ValidationError

from rag_ops_guard.app import conversation_agent
from rag_ops_guard.configstore.runtime import EffectiveConfig, resolve_effective_config
from rag_ops_guard.diagnostics.connectivity import probe_runtime_connectivity
from rag_ops_guard.domain.models import QueryRequest, QueryResponse, ResponseOutcome
from rag_ops_guard.observability.runtime import (
    QUERY_LOGGER,
    QUERY_METRICS,
    add_count,
    add_milliseconds,
    add_seconds,
    config_observability_fields,
    set_config_dimensions,
)


def _internal_error_body(exc: Exception) -> str:
    payload: dict[str, str] = {"error": "internal_error"}
    if os.environ.get("APP_ENV") == "local":
        payload["detail"] = str(exc)
    return json.dumps(payload)


def _proxy_response(response: QueryResponse) -> dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": response.model_dump_json(),
    }


def _json_proxy(status_code: int, payload: dict[str, object]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def _connectivity_probe_response(effective: EffectiveConfig) -> dict[str, Any]:
    if os.environ.get("APP_ENV") != "local":
        return _json_proxy(404, {"error": "not_found"})
    result = probe_runtime_connectivity(effective)
    payload: dict[str, object] = {
        "probe": "runtime_connectivity",
        "function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "rag-ops-guard-query"),
        **result,
    }
    return _json_proxy(200 if bool(result.get("ok")) else 503, payload)


def _record_config_observability(
    effective: EffectiveConfig, resolve_latency_ms: float
) -> dict[str, object]:
    fields = config_observability_fields(
        revision_no=effective.revision_no,
        config_hash=effective.config_hash,
        source=effective.source,
    )
    set_config_dimensions(
        QUERY_METRICS,
        revision_no=effective.revision_no,
        config_hash=effective.config_hash,
        source=effective.source,
    )
    add_milliseconds(QUERY_METRICS, "ConfigResolveLatencyMs", resolve_latency_ms)
    if effective.source == "cache":
        add_count(QUERY_METRICS, "ConfigCacheHit")
    elif effective.source != "env":
        add_count(QUERY_METRICS, "ConfigCacheMiss")
    if effective.db_unavailable:
        add_count(QUERY_METRICS, "ConfigDbUnavailable")
    if effective.stale:
        add_count(QUERY_METRICS, "ConfigStaleServed")
    if effective.revision_age_s is not None:
        add_seconds(QUERY_METRICS, "ConfigRevisionAge", effective.revision_age_s)

    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.metadata.update(fields)
    return dict(fields)


@traceable(name="rag_query", run_type="chain")
def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    del context
    started = perf_counter()
    config_fields: dict[str, object] = {
        "config_revision": None,
        "config_hash": "unknown",
        "config_source": "unresolved",
    }
    set_config_dimensions(
        QUERY_METRICS,
        revision_no=None,
        config_hash="unknown",
        source="unresolved",
    )
    try:
        resolve_started = perf_counter()
        effective = resolve_effective_config()
        config_fields = _record_config_observability(
            effective,
            (perf_counter() - resolve_started) * 1000,
        )
        add_count(QUERY_METRICS, "QueryCount")

        body = event.get("body", event)
        payload = json.loads(body) if isinstance(body, str) else body
        if isinstance(payload, dict) and payload.get("__rag_ops_probe__") == "connectivity":
            return _connectivity_probe_response(effective)

        request = QueryRequest.model_validate(payload)
        if effective.fail_closed:
            add_count(QUERY_METRICS, "ConfigFailClosed")
            response = QueryResponse(
                request_id="config-fail-closed",
                outcome=ResponseOutcome.INSUFFICIENT_EVIDENCE,
                message=(
                    "Configuration safety policy is older than the permitted stale window; "
                    "document-backed claims are disabled until a trusted configuration source "
                    "is available."
                ),
                route="uncertain",
                config_hash=effective.config_hash,
            )
            QUERY_LOGGER.warning(
                "config_fail_closed",
                extra={
                    **config_fields,
                    "stale_age_s": effective.stale_age_s,
                },
            )
            return _proxy_response(response)

        with tracing_context(metadata=config_fields):
            response = conversation_agent().invoke(request).model_copy(
                update={"config_hash": effective.config_hash}
            )
        add_count(QUERY_METRICS, f"{response.status.value.title().replace('_', '')}Count")
        QUERY_LOGGER.info(
            "query_completed",
            extra={
                **config_fields,
                "request_id": response.request_id,
                "status": response.status.value,
                "citation_count": len(response.citations),
                "thread_id": request.thread_id,
                "config_stale": effective.stale,
            },
        )
        return _proxy_response(response)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        add_count(QUERY_METRICS, "InvalidRequestCount")
        QUERY_LOGGER.warning(
            "invalid_query_request",
            extra={**config_fields, "error_type": type(exc).__name__},
        )
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_request", "detail": str(exc)}),
        }
    except Exception as exc:
        add_count(QUERY_METRICS, "InternalErrorCount")
        QUERY_LOGGER.exception(
            "query_failed",
            extra={**config_fields, "error_type": type(exc).__name__},
        )
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": _internal_error_body(exc),
        }
    finally:
        add_milliseconds(QUERY_METRICS, "QueryLatencyMs", (perf_counter() - started) * 1000)
        QUERY_METRICS.flush_metrics()
