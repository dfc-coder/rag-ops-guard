from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_ops_guard.domain.models import IngestResponse
from rag_ops_guard.handlers import ingest
from rag_ops_guard.tenancy import RequestContext

_CONTEXT = RequestContext(principal="api-key:key-a", tenant_id="tenant-a")


class _FailingIngestionService:
    def ingest(self, s3_key: str) -> object:
        del s3_key
        raise RuntimeError("vector backend unavailable")


class _SuccessfulIngestionService:
    def ingest(self, s3_key: str) -> IngestResponse:
        del s3_key
        return IngestResponse(
            status="no_op",
            document_id="doc-1",
            logical_id="logical-1",
            version="1",
            chunks=2,
        )


def _effective(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "config_hash": "12345678" + "0" * 56,
        "source": "dynamodb",
        "revision_no": 9,
        "stale": False,
        "stale_age_s": 0.0,
        "fail_closed": False,
        "db_unavailable": False,
        "revision_age_s": 8.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quiet_observability(monkeypatch) -> tuple[MagicMock, MagicMock, list[str], list[tuple[str, float]]]:
    logger = MagicMock()
    metrics = MagicMock()
    counts: list[str] = []
    measures: list[tuple[str, float]] = []
    monkeypatch.setattr(ingest, "INGEST_LOGGER", logger)
    monkeypatch.setattr(ingest, "INGEST_METRICS", metrics)
    monkeypatch.setattr(
        ingest,
        "add_count",
        lambda _metrics, name, value=1.0: counts.append(name),
    )
    monkeypatch.setattr(
        ingest,
        "add_milliseconds",
        lambda _metrics, name, value: measures.append((name, value)),
    )
    monkeypatch.setattr(
        ingest,
        "add_seconds",
        lambda _metrics, name, value: measures.append((name, value)),
        raising=False,
    )
    monkeypatch.setattr(
        ingest,
        "resolve_tenant_effective_config",
        lambda tenant_id: _effective(),
        raising=False,
    )
    monkeypatch.setattr(ingest, "request_context_from_event", lambda event: _CONTEXT)
    return logger, metrics, counts, measures


def test_unexpected_ingest_failure_returns_diagnostic_500_in_local(monkeypatch) -> None:
    logger, _, _, _ = _quiet_observability(monkeypatch)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setattr(ingest, "ingestion_service", lambda context: _FailingIngestionService())

    response = ingest.handler(
        {"body": json.dumps({"s3_key": "t/tenant-a/raw/test.md"})},
        object(),
    )

    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {
        "error": "internal_error",
        "detail": "vector backend unavailable",
    }
    logger.exception.assert_called_once()


def test_phase2_ingest_emits_required_config_dimensions_and_health_metrics(monkeypatch) -> None:
    logger, metrics, counts, measures = _quiet_observability(monkeypatch)
    monkeypatch.setattr(ingest, "ingestion_service", lambda context: _SuccessfulIngestionService())

    response = ingest.handler(
        {"body": json.dumps({"s3_key": "t/tenant-a/raw/test.md"})},
        object(),
    )

    assert response["statusCode"] == 200
    metrics.add_dimension.assert_any_call(name="config_revision", value="9")
    metrics.add_dimension.assert_any_call(name="config_hash", value="12345678")
    metrics.add_dimension.assert_any_call(name="config_source", value="dynamodb")
    assert "ConfigCacheMiss" in counts
    assert "ConfigDbUnavailable" not in counts
    assert any(name == "ConfigResolveLatencyMs" for name, _ in measures)
    assert ("ConfigRevisionAge", 8.0) in measures
    log_extra = logger.info.call_args.kwargs["extra"]
    assert log_extra["config_revision"] == 9
    assert log_extra["config_hash"] == "12345678"
    assert log_extra["config_source"] == "dynamodb"
    assert log_extra["tenant_id"] == "tenant-a"
