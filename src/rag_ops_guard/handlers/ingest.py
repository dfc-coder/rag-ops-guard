from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from rag_ops_guard.app import ingestion_service
from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import IngestRequest
from rag_ops_guard.observability.runtime import (
    INGEST_LOGGER,
    INGEST_METRICS,
    add_count,
    add_milliseconds,
)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    del context
    started = perf_counter()
    add_count(INGEST_METRICS, "IngestRequestCount")
    try:
        body = event.get("body", event)
        payload = json.loads(body) if isinstance(body, str) else body
        request = IngestRequest.model_validate(payload)
        response = ingestion_service().ingest(request.s3_key)
        add_count(INGEST_METRICS, "IngestedCount" if response.status == "ingested" else "NoOpCount")
        INGEST_LOGGER.info(
            "ingest_completed",
            extra={
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
    except DocumentValidationError as exc:
        add_count(INGEST_METRICS, "InvalidDocumentCount")
        INGEST_LOGGER.warning("invalid_document", extra={"error_type": type(exc).__name__})
        return {
            "statusCode": 422,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_document", "detail": str(exc)}),
        }
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        add_count(INGEST_METRICS, "InvalidRequestCount")
        INGEST_LOGGER.warning("invalid_ingest_request", extra={"error_type": type(exc).__name__})
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "invalid_request", "detail": str(exc)}),
        }
    finally:
        add_milliseconds(INGEST_METRICS, "IngestLatencyMs", (perf_counter() - started) * 1000)
        INGEST_METRICS.flush_metrics()
