from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from rag_ops_guard.app import conversation_agent
from rag_ops_guard.domain.models import QueryRequest
from rag_ops_guard.observability.runtime import (
    QUERY_LOGGER,
    QUERY_METRICS,
    add_count,
    add_milliseconds,
)


def _internal_error_body(exc: Exception) -> str:
    payload: dict[str, str] = {"error": "internal_error"}
    if os.environ.get("APP_ENV") == "local":
        payload["detail"] = str(exc)
    return json.dumps(payload)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    del context
    started = perf_counter()
    add_count(QUERY_METRICS, "QueryCount")
    try:
        body = event.get("body", event)
        payload = json.loads(body) if isinstance(body, str) else body
        request = QueryRequest.model_validate(payload)
        response = conversation_agent().invoke(request)
        add_count(QUERY_METRICS, f"{response.status.value.title().replace('_', '')}Count")
        QUERY_LOGGER.info(
            "query_completed",
            extra={
                "request_id": response.request_id,
                "status": response.status.value,
                "citation_count": len(response.citations),
            },
        )
        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": response.model_dump_json(),
        }
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        add_count(QUERY_METRICS, "InvalidRequestCount")
        QUERY_LOGGER.warning("invalid_query_request", extra={"error_type": type(exc).__name__})
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_request", "detail": str(exc)}),
        }
    except Exception as exc:
        add_count(QUERY_METRICS, "InternalErrorCount")
        QUERY_LOGGER.exception("query_failed", extra={"error_type": type(exc).__name__})
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": _internal_error_body(exc),
        }
    finally:
        add_milliseconds(QUERY_METRICS, "QueryLatencyMs", (perf_counter() - started) * 1000)
        QUERY_METRICS.flush_metrics()
